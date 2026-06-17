"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re

import decord
from decord import VideoReader
import pandas as pd
import numpy as np
import torch

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset

decord.bridge.set_bridge("torch")

class LVUCLSDataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, 
                 history, num_frames, task, stride=10, split='train'):
        """
        vis_root (string): Root directory of videos (e.g. LVU/videos/)
        ann_root (string): directory to store the gt_dict file
        """
        self.vis_root = vis_root
        
        task_list = ['director', 'genre', 'relationship', 'scene', 'way_speaking', 'writer', 'year']
        assert task in task_list, f'Invalid task {task}, must be one of {task_list}'
        self.task = task

        self.gt_dict = {}
        for ann_path in ann_paths:
            self.gt_dict.update(json.load(open(ann_path)))

        self.fps = 10
        self.annotation = {}
        self.stride = stride
        for video_id in self.gt_dict:
            if task in self.gt_dict[video_id]:
                duration = self.gt_dict[video_id]['duration']
                video_length = self.gt_dict[video_id]['num_frames']
                label = self.gt_dict[video_id][task]
                label_after_process = text_processor(label)
                assert label == label_after_process, "{} not equal to {}".format(label, label_after_process)
                self.annotation[f'{video_id}_0'] = {'video_id': video_id, 'start': 0, 'label': label_after_process, 'duration': duration, 'video_length': video_length, 'answer': self.gt_dict[video_id][f'{task}_answer']}
                for start in range(self.stride, duration - history + 1, self.stride):
                    video_start_id = f'{video_id}_{start}'
                    self.annotation[video_start_id] = {'video_id': video_id, 'start': start, 'label': label_after_process, 'duration': duration, 'video_length': video_length, 'answer': self.gt_dict[video_id][f'{task}_answer']}
        
        self.data_list = list(self.annotation.keys())
        self.data_list.sort()

        # Filter out missing videos and track skipped items
        original_len = len(self.data_list)
        self.data_list = [vid for vid in self.data_list if self._video_exists(self.annotation[vid]['video_id'])]
        self.skipped_count = original_len - len(self.data_list)
        if self.skipped_count > 0:
            import logging
            logger = logging.getLogger("lavis.datasets")
            logger.info(f"Skipped {self.skipped_count} missing video files out of {original_len} samples")

        self.history = history
        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]['start']
        end_time = min(self.annotation[video_start_id]['start'] + self.history - 1, self.annotation[video_start_id]['duration'])

        video_id = self.annotation[video_start_id]['video_id']
        video_path = self._find_video_path(video_id)
        vr = VideoReader(uri=video_path, ctx=decord.gpu(0))
        fps = vr.get_avg_fps()
        start_frame_index = int(start_time * fps)
        end_frame_index = min(int(end_time * fps), len(vr) - 1)
        selected_frame_index = np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames)).astype(int).tolist()
        # (T, H, W, C) -> (C, T, H, W)
        video = vr.get_batch(selected_frame_index).permute(3, 0, 1, 2).float()
        video = self.vis_processor(video)

        text_input = self.text_processor(f'what is the {self.task} of the movie?')
        caption = self.text_processor(self.annotation[video_start_id]['label'])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }

    def _video_exists(self, video_id):
        for ext in ('.mp4', '.mkv', '.avi', '.webm'):
            path = os.path.join(self.vis_root, video_id + ext)
            if os.path.exists(path):
                return True
        return False

    def _find_video_path(self, video_id):
        for ext in ('.mp4', '.mkv', '.avi', '.webm'):
            path = os.path.join(self.vis_root, video_id + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No video file found for {video_id} under {self.vis_root}")

    def __len__(self):
        return len(self.data_list)

class LVUCLSEvalDataset(LVUCLSDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, 
                 history, num_frames, task, stride=10, split='val'):
        
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, 
                 history, num_frames, task, stride=stride, split=split)

    def __getitem__(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]['start']
        end_time = min(self.annotation[video_start_id]['start'] + self.history - 1, self.annotation[video_start_id]['duration'])

        video_id = self.annotation[video_start_id]['video_id']
        video_path = self._find_video_path(video_id)
        vr = VideoReader(uri=video_path, ctx=decord.gpu(0))
        fps = vr.get_avg_fps()
        start_frame_index = int(start_time * fps)
        end_frame_index = min(int(end_time * fps), len(vr) - 1)
        selected_frame_index = np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames)).astype(int).tolist()
        # (T, H, W, C) -> (C, T, H, W)
        video = vr.get_batch(selected_frame_index).permute(3, 0, 1, 2).float()
        video = self.vis_processor(video)

        text_input = self.text_processor(f'what is the {self.task} of the movie?')
        caption = self.text_processor(self.annotation[video_start_id]['label'])
        return {
            "image": video,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }

