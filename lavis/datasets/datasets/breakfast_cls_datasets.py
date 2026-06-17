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

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset

decord.bridge.set_bridge("torch")

class BreakfastCLSDataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.gt_dict = {}
        for ann_path in ann_paths:
            self.gt_dict.update(json.load(open(ann_path)))

        self.fps = 10
        self.annotation = {}
        for video_id in self.gt_dict:
            if video_id in ['P28-cam01-P28_cereals', 'P27-stereo-P27_milk_ch0', 'P28-cam02-P28_cereals']:
                continue
            label = self.gt_dict[video_id]['class_name']
            label_after_process = text_processor(label)
            assert label == label_after_process, "{} not equal to {}".format(label, label_after_process)
            self.annotation[video_id] = {'video_id': video_id, 'label': label_after_process}

        self.video_id_list = list(self.annotation.keys())
        self.video_id_list.sort()
        self.skipped_count = 0

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt

    def _parse_video_id(self, video_id):
        parts = video_id.split('-')
        person = parts[0]
        camera = parts[1]
        action = parts[2]
        return person, camera, action

    def _video_exists(self, video_id):
        person, camera, action = self._parse_video_id(video_id)
        video_dir = os.path.join(self.vis_root, person, camera)
        for ext in ('.avi', '.mp4', '.mkv', '.webm'):
            if os.path.exists(os.path.join(video_dir, action + ext)):
                return True
        return False

    def _find_video_path(self, video_id):
        person, camera, action = self._parse_video_id(video_id)
        video_dir = os.path.join(self.vis_root, person, camera)
        for ext in ('.avi', '.mp4', '.mkv', '.webm'):
            path = os.path.join(video_dir, action + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No video file found for {video_id} under {video_dir}")

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]

        try:
            video_path = self._find_video_path(video_id)
        except FileNotFoundError:
            self.skipped_count += 1
            logging.getLogger("lavis.datasets").warning(
                f"Skipped missing video {video_id} (total skipped: {self.skipped_count})"
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

        text_input = self.text_processor('what type of breakfast is shown in the video?')
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_id,
        }

    def __len__(self):
        return len(self.video_id_list)

class BreakfastCLSEvalDataset(BreakfastCLSDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 num_frames, prompt, split='val'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='val')

    def __getitem__(self, index):
        video_id = self.video_id_list[index]
        ann = self.annotation[video_id]

        try:
            video_path = self._find_video_path(video_id)
        except FileNotFoundError:
            self.skipped_count += 1
            logging.getLogger("lavis.datasets").warning(
                f"Skipped missing video {video_id} (total skipped: {self.skipped_count})"
            )
            return self.__getitem__((index + 1) % len(self))

        vr = VideoReader(uri=video_path, ctx=decord.gpu(0))
        total_frames = len(vr)

        # Uniform frame sampling (eval)
        selected_frame_index = np.rint(np.linspace(0, total_frames - 1, self.num_frames)).astype(int).tolist()
        video = vr.get_batch(selected_frame_index).permute(3, 0, 1, 2).float()
        video = self.vis_processor(video)

        text_input = self.text_processor('what type of breakfast is shown in the video?')
        caption = self.text_processor(ann['label'])
        return {
            "image": video,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_id,
        }
