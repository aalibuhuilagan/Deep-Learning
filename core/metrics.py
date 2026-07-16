# -*- coding: utf-8 -*-
# 进阶：量化评估指标

import torch

def calculate_confidence_change(model, img_tensor, mask):
    """计算遮挡后的置信度变化"""
    with torch.no_grad():
        original_out = model(img_tensor)
        original_prob = torch.softmax(original_out, dim=1)
        top_idx = original_prob.argmax().item()
        top_prob = original_prob[0][top_idx].item()

        masked_tensor = img_tensor * (1 - mask)
        masked_out = model(masked_tensor)
        masked_prob = torch.softmax(masked_out, dim=1)[0][top_idx].item()

        drop = top_prob - masked_prob
        return top_idx, top_prob, masked_prob, drop