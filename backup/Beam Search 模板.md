```python

import torch
import torch.nn.functional as F

class Beam:
    def __init__(self, token, logp, h, sequence):
        # 初始化beam对象
        self.token = token  # 当前token
        self.logp = logp  # 当前token的对数概率
        self.h = h  # 隐藏状态
        self.sequence = sequence  # 到目前为止的序列
        self.done = (token == VOCAB+1)  # 判断是否到达序列结束标记EOS

    def extend(self, token, logp, h):
        # 扩展当前beam，返回新的beam实例
        return Beam(token, self.logp + logp, h, self.sequence + [token])


def beam_search(model, inp, beam_width=5, max_decoding_len=15, top_k=10):
    # 使用beam search算法进行解码
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        inp = inp.to(device)  # 将输入移到设备上
        out_enc, h = model.enc(model.emb(inp))  # 编码器部分
        out_enc = model.DP(out_enc)  # 应用dropout
        h = h.view((1, inp.shape[0], 2 * model.HID))  # 调整隐藏状态的形状

        start_token = char_to_idx['begin']  # 开始token
        beam = [Beam(start_token, 0.0, h, [start_token])]  # 初始化beam列表

        for _ in range(max_decoding_len):  # 最大解码长度
            new_beam = []

            for b in beam:
                if b.done:
                    new_beam.append(b)  # 如果完成则直接添加到新beam列表
                    continue

                dec_inp = torch.tensor([[b.token]], dtype=torch.long, device=device)  # 准备解码器的输入
                dec_out, h_new = model.run_dec(dec_inp, out_enc, b.h)  # 解码器运行
                log_probs = F.log_softmax(dec_out, dim=-1)  # 使用log_softmax获取概率

                top_k_log_probs, top_k_tokens = torch.topk(log_probs, beam_width)  # 获取top_k概率和对应的tokens

                for i in range(beam_width):
                    token = top_k_tokens[0, 0, i].item()  # 获取token
                    logp = top_k_log_probs[0, 0, i].item()  # 获取对数概率
                    new_beam.append(b.extend(token, logp, h_new))  # 扩展beam

            beam = sorted(new_beam, key=lambda x: x.logp / len(x.sequence), reverse=True)[:beam_width]  # 排序并保留最好的beam_width个beam

        # 返回分数最高的top_k个序列
        return [b.sequence for b in sorted(beam, key=lambda x: x.logp / len(x.sequence), reverse=True)[:top_k]]
```