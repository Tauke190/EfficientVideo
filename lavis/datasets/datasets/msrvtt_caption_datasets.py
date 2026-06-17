"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import logging

import decord
from decord import VideoReader
import numpy as np
import torch

from lavis.datasets.datasets.video_caption_datasets import VideoCaptionDataset

decord.bridge.set_bridge("torch")

class MSRVTTCapDataset(VideoCaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.video_id_list = list(self.annotation.keys())
        self.video_id_list.sort()
        self.fps = 10
        self.skipped_count = 0

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt

    def _find_video_path(self, video_id):
        for ext in ('.mp4', '.mkv', '.avi', '.webm'):
            path = os.path.join(self.vis_root, video_id + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No video file found for {video_id} under {self.vis_root}")

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]
        video_name = ann['video']

        try:
            video_path = self._find_video_path(video_name)
        except FileNotFoundError:
            self.skipped_count += 1
            logging.getLogger("lavis.datasets").warning(
                f"Skipped missing video {video_name} (total skipped: {self.skipped_count})"
            )
            return self.__getitem__((index + 1) % len(self))

        vr = VideoReader(uri=video_path, ctx=decord.gpu(0))
        total_frames = len(vr)

        # Random segment sampling (train)
        segment_list = np.linspace(0, total_frames, self.num_frames + 1, dtype=int)
        selected_frame_index = []
        for start, end in zip(segment_list[:-1], segment_list[1:]):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        video = vr.get_batch(selected_frame_index).permute(3, 0, 1, 2).float()
        video = self.vis_processor(video)

        text_input = self.prompt
        caption = self.text_processor.pre_caption(ann["caption"])

        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "prompt": self.prompt,
            "image_id": ann["video"],
        }

    def __len__(self):
        return len(self.video_id_list)

class MSRVTTCapEvalDataset(MSRVTTCapDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val')

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]
        video_name = ann['video']

        try:
            video_path = self._find_video_path(video_name)
        except FileNotFoundError:
            self.skipped_count += 1
            logging.getLogger("lavis.datasets").warning(
                f"Skipped missing video {video_name} (total skipped: {self.skipped_count})"
            )
            return self.__getitem__((index + 1) % len(self))

        vr = VideoReader(uri=video_path, ctx=decord.gpu(0))
        total_frames = len(vr)

        # Uniform frame sampling (eval)
        selected_frame_index = np.rint(np.linspace(0, total_frames - 1, self.num_frames)).astype(int).tolist()
        video = vr.get_batch(selected_frame_index).permute(3, 0, 1, 2).float()
        video = self.vis_processor(video)

        text_input = self.prompt
        caption = self.text_processor.pre_caption(ann["caption"])

        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "prompt": self.prompt,
            "image_id": ann["video"],
        }
