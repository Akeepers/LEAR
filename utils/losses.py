import torch.nn.functional as F
import torch
# define the loss function


def position_loss_v1(golden, infer, input_mask, is_logit=True):
    label_num = infer.shape[-1]
    label_mask = input_mask.unsqueeze(-1).expand(-1, -1, label_num)

    loss = F.binary_cross_entropy_with_logits(
        infer.view(-1), golden.view(-1), reduction="none") if is_logit else F.binary_cross_entropy(infer.view(-1), golden.view(-1), reduction="none")
    loss = (loss * label_mask.contiguous().view(-1)).sum() / label_mask.sum()
    return loss


def span_loss_v1(golden, infer, input_mask, loss_type='golden_and_infer', span_candidate=None, is_logit=True):
    if loss_type == "golden_and_infer":
        assert span_candidate is not None
    batch_size, _, seq_len, label_num = infer.size()
    label_mask = input_mask.unsqueeze(-1).expand(-1, -1, label_num)
    span_label_mask = label_mask.bool().unsqueeze(
        2).expand(-1, -1, seq_len, -1) & label_mask.bool().unsqueeze(1).expand(-1, seq_len, -1, -1)
    # start should be less equal to end
    span_label_mask = torch.triu(span_label_mask.permute(
        0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
        batch_size, label_num, seq_len, seq_len).permute(0, 2, 3, 1)
    if loss_type == "golden_and_infer":
        span_label_mask = span_label_mask & span_candidate
    float_span_label_mask = span_label_mask.contiguous().view(batch_size, -1).float()
    span_loss = F.binary_cross_entropy_with_logits(
        infer.contiguous().view(batch_size, -1), golden.contiguous().view(batch_size, -1), reduction="none") if is_logit else F.binary_cross_entropy(infer.contiguous().view(batch_size, -1), golden.contiguous().view(batch_size, -1), reduction="none")
    span_loss = span_loss * float_span_label_mask
    span_loss = span_loss.sum() / (float_span_label_mask.sum())
    return span_loss


def position_loss_v2(golden, infer, input_mask, is_logit=True):
    label_num = infer.shape[-1]
    label_mask = input_mask.unsqueeze(-1).expand(-1, -1, label_num)

    loss = F.binary_cross_entropy_with_logits(
        infer.view(-1), golden.view(-1), reduction="none") if is_logit else F.binary_cross_entropy(infer.view(-1), golden.view(-1), reduction="none")
    loss = loss * label_mask.contiguous().view(-1)
    loss = torch.sum(loss.contiguous().view(-1, label_num),
                     dim=-1).sum() / (label_mask.sum()/label_num + 1e-10)
    return loss


def span_loss_v2(golden, infer, input_mask, loss_type='golden_and_infer', span_candidate=None, is_logit=True):
    if loss_type == "golden_and_infer":
        assert span_candidate is not None
    batch_size, _, seq_len, label_num = infer.size()
    label_mask = input_mask.unsqueeze(-1).expand(-1, -1, label_num)
    span_label_mask = label_mask.bool().unsqueeze(
        2).expand(-1, -1, seq_len, -1) & label_mask.bool().unsqueeze(1).expand(-1, seq_len, -1, -1)
    # start should be less equal to end
    # span_label_mask = torch.triu(span_label_mask.permute(
    #     0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
    #     batch_size, label_num, seq_len, seq_len).permute(0, 2, 3, 1)
    span_label_mask = torch.triu(span_label_mask.transpose(
        1, 3).transpose(2, 3).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
        batch_size, label_num, seq_len, seq_len).transpose(1, 2).transpose(2, 3)
    if loss_type == "golden_and_infer":
        span_label_mask = span_label_mask & span_candidate
    float_span_label_mask = span_label_mask.contiguous().view(batch_size, -1).float()
    span_loss = F.binary_cross_entropy_with_logits(
        infer.contiguous().view(batch_size, -1), golden.contiguous().view(batch_size, -1), reduction="none") if is_logit else F.binary_cross_entropy(infer.contiguous().view(batch_size, -1), golden.contiguous().view(batch_size, -1), reduction="none")
    span_loss = span_loss * float_span_label_mask
    span_loss = torch.sum(span_loss.contiguous(
    ).view(-1, label_num), dim=-1).sum() / (float_span_label_mask.sum() / label_num + 1e-10)
    return span_loss


def _multilabel_categorical_crossentropy(y_pred, y_true, ghm=True):
    """
    y_pred: (batch_size, shaking_seq_len, type_size)
    y_true: (batch_size, shaking_seq_len, type_size)
    y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
         1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred oudtuts of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred oudtuts of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])  # st - st
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def compute_focal_loss(bce_loss, labels, alpha=0.25, gamma=2):
    # """
    # focal_loss损失计算
    # :param infers:  预测结果. size:[B,seq_len,2]
    # :param labels:  实际类别. size:[B,seq_len]
    # :return:
    # """
    pt = torch.exp(-bce_loss)
    pos_alpha = (1-alpha) * labels
    neg_alpha = alpha * (~labels.bool()).float()
    alpha = pos_alpha + neg_alpha
    focal_loss = (1-pt)**gamma*bce_loss
    # print(focal_loss.shape)
    # print(alpha.view(-1).shape)
    if len(bce_loss.shape) == 2:
        alpha = alpha.view(bce_loss.shape)
    else:
        alpha = alpha.view(-1)
    focal_loss = torch.mul(focal_loss, alpha)
    return focal_loss


def compute_focal_loss_v2(start_golden, start_infer, end_golden, end_infer, span_golden, span_infer, input_mask, loss_type='golden_and_infer', is_logit=True, alpha=0.25, gamma=2):
    batch_size, seq_len, label_num = start_infer.size()
    if loss_type == "golden_and_infer":
        start_infer_ = start_infer > 0
        end_infer_ = end_infer > 0
        span_candidate = torch.logical_or(
            (start_infer_.unsqueeze(-2).expand(-1, -1, seq_len, -1)
                & end_infer_.unsqueeze(-3).expand(-1, seq_len, -1, -1)),
            (start_golden.unsqueeze(-2).expand(-1, -1, seq_len, -1).bool()
                & end_infer.unsqueeze(-3).expand(-1, seq_len, -1, -1).bool())
        )

    label_mask = input_mask.unsqueeze(-1).expand(-1, -1, label_num)

    start_loss = F.binary_cross_entropy_with_logits(
        start_infer.view(-1), start_golden.view(-1), reduction="none") if is_logit else F.binary_cross_entropy(start_infer.view(-1), start_golden.view(-1), reduction="none")
    start_loss = compute_focal_loss(start_loss, start_golden, alpha, gamma)
    start_loss = start_loss * label_mask.contiguous().view(-1)
    # start_loss = torch.sum(start_loss.contiguous().view(-1, label_num),
    #                        dim=-1).sum() / (label_mask.sum()/label_num + 1e-10)
    start_loss = torch.sum(start_loss.contiguous().view(-1),
                           dim=-1).sum() / (label_mask.sum() + 1e-10)

    end_loss = F.binary_cross_entropy_with_logits(
        end_infer.view(-1), end_golden.view(-1), reduction="none") if is_logit else F.binary_cross_entropy(end_infer.view(-1), end_golden.view(-1), reduction="none")
    end_loss = compute_focal_loss(end_loss, end_golden, alpha, gamma)
    end_loss = end_loss * label_mask.contiguous().view(-1)
    # end_loss = torch.sum(end_loss.contiguous().view(-1, label_num),
    #                      dim=-1).sum() / (label_mask.sum()/label_num + 1e-10)

    end_loss = torch.sum(end_loss.contiguous().view(-1), dim=-1).sum() / (label_mask.sum() + 1e-10)

    span_label_mask = label_mask.bool().unsqueeze(
        2).expand(-1, -1, seq_len, -1) & label_mask.bool().unsqueeze(1).expand(-1, seq_len, -1, -1)

    # start should be less equal to end
    span_label_mask = torch.triu(span_label_mask.transpose(
        1, 3).transpose(2, 3).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
        batch_size, label_num, seq_len, seq_len).transpose(1, 2).transpose(2, 3)
    if loss_type == "golden_and_infer":
        span_label_mask = span_label_mask & span_candidate
    float_span_label_mask = span_label_mask.contiguous().view(batch_size, -1).float()
    span_loss = F.binary_cross_entropy_with_logits(
        span_infer.contiguous().view(batch_size, -1), span_golden.contiguous().view(batch_size, -1), reduction="none") if is_logit else F.binary_cross_entropy(span_infer.contiguous().view(batch_size, -1), span_golden.contiguous().view(batch_size, -1), reduction="none")
    span_loss = span_loss * float_span_label_mask
    span_loss = compute_focal_loss(span_loss, span_golden, alpha, gamma)
    # span_loss = torch.sum(span_loss.contiguous().view(-1, label_num),
    #                       dim=-1).sum() / (float_span_label_mask.sum() / label_num + 1e-10)
    span_loss = torch.sum(span_loss.contiguous().view(-1),
                          dim=-1).sum() / (float_span_label_mask.sum() + 1e-10)

    return start_loss, end_loss, span_loss


def compute_loss_v2(start_golden, start_infer, end_golden, end_infer, span_golden, span_infer, input_mask, loss_type='golden_and_infer', is_logit=True, loss_reweight=-1):
    batch_size, seq_len, label_num = start_infer.size()
    if loss_type == "golden_and_infer":
        start_infer_ = start_infer > 0
        end_infer_ = end_infer > 0
        span_candidate = torch.logical_or(
            (start_infer_.unsqueeze(-2).expand(-1, -1, seq_len, -1)
                & end_infer_.unsqueeze(-3).expand(-1, seq_len, -1, -1)),
            (start_golden.unsqueeze(-2).expand(-1, -1, seq_len, -1).bool()
                & end_infer.unsqueeze(-3).expand(-1, seq_len, -1, -1).bool())
        )

    # if loss_reweight != -1:
    #     if is_logit:
    #         start_infer = tor

    label_mask = input_mask.unsqueeze(-1).expand(-1, -1, label_num)

    start_loss = F.binary_cross_entropy_with_logits(
        start_infer.view(-1), start_golden.view(-1), reduction="none") if is_logit else F.binary_cross_entropy(start_infer.view(-1), start_golden.view(-1), reduction="none")
    start_loss = start_loss * label_mask.contiguous().view(-1)
    start_loss = torch.sum(start_loss.contiguous().view(-1, label_num),
                           dim=-1).sum() / (label_mask.sum()/label_num + 1e-10)
    # start_loss = torch.sum(start_loss.contiguous().view(-1),
    #                        dim=-1).sum() / (label_mask.sum() + 1e-10)

    end_loss = F.binary_cross_entropy_with_logits(
        end_infer.view(-1), end_golden.view(-1), reduction="none") if is_logit else F.binary_cross_entropy(end_infer.view(-1), end_golden.view(-1), reduction="none")
    end_loss = end_loss * label_mask.contiguous().view(-1)
    end_loss = torch.sum(end_loss.contiguous().view(-1, label_num),
                         dim=-1).sum() / (label_mask.sum()/label_num + 1e-10)

    # end_loss = torch.sum(end_loss.contiguous().view(-1), dim=-1).sum() / (label_mask.sum() + 1e-10)

    span_label_mask = label_mask.bool().unsqueeze(
        2).expand(-1, -1, seq_len, -1) & label_mask.bool().unsqueeze(1).expand(-1, seq_len, -1, -1)
    # start should be less equal to end
    # span_label_mask = torch.triu(span_label_mask.permute(
    #     0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
    #     batch_size, label_num, seq_len, seq_len).permute(0, 2, 3, 1)

    # start should be less equal to end
    span_label_mask = torch.triu(span_label_mask.transpose(
        1, 3).transpose(2, 3).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
        batch_size, label_num, seq_len, seq_len).transpose(1, 2).transpose(2, 3)
    if loss_type == "golden_and_infer":
        span_label_mask = span_label_mask & span_candidate
    float_span_label_mask = span_label_mask.contiguous().view(batch_size, -1).float()
    span_loss = F.binary_cross_entropy_with_logits(
        span_infer.contiguous().view(batch_size, -1), span_golden.contiguous().view(batch_size, -1), reduction="none") if is_logit else F.binary_cross_entropy(span_infer.contiguous().view(batch_size, -1), span_golden.contiguous().view(batch_size, -1), reduction="none")
    span_loss = span_loss * float_span_label_mask
    span_loss = torch.sum(span_loss.contiguous().view(-1, label_num),
                          dim=-1).sum() / (float_span_label_mask.sum() / label_num + 1e-10)
    # span_loss = torch.sum(span_loss.contiguous().view(-1),
    #                       dim=-1).sum() / (float_span_label_mask.sum() + 1e-10)

    # span_loss = torch.sum(span_loss.contiguous().view(-1, label_num),
    #                       dim=-1).sum() / (float_span_label_mask.sum() + 1e-10)
    return start_loss, end_loss, span_loss


def loss_v1(gold, pred, mask):
    pred = pred.squeeze(-1)
    gold = gold.cuda()
    mask = mask.cuda()

    los = F.binary_cross_entropy(pred, gold, reduction='none')
    if los.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    tmp = los * mask
    los = torch.sum(los * mask) / torch.sum(mask)
    return los


def loss_v2(gold, infer, padding_mask, loss_mask=None, is_logit=True, weight=-1):
    if loss_mask is not None:
        loss_mask = loss_mask.view(-1, 1, 1)
        infer = infer * loss_mask
        gold = gold * loss_mask
    label_num = infer.shape[-1]
    # print(infer.shape)
    # print(padding_mask.shape)
    active_pos = padding_mask.contiguous().view(-1) == 1
    masked_infer = infer.contiguous().view(-1, label_num)[active_pos]
    masked_gold = gold.contiguous().view(-1, label_num)[active_pos]
    loss_ = F.binary_cross_entropy_with_logits(masked_infer, masked_gold, reduction='none') if is_logit else F.binary_cross_entropy(
        masked_infer, masked_gold, reduction='none')
    loss_ = torch.sum(loss_, 1)
    loss = torch.mean(loss_)
    return loss


def loss_v3(gold, infer, padding_mask, loss_mask=None):
    if loss_mask is not None:
        loss_mask = loss_mask.view(-1, 1, 1)
        infer = infer * loss_mask
        gold = gold * loss_mask
    label_num = infer.shape[-1]
    active_pos = padding_mask.view(-1) == 1
    masked_infer = infer.view(-1, label_num)[active_pos]
    masked_gold = gold.view(-1, label_num)[active_pos]
    loss_ = F.binary_cross_entropy(
        masked_infer, masked_gold, reduction='none')
    loss_ = torch.sum(loss_, 1)
    loss = torch.sum(loss_)
    #     batch_size = infer.shape[0]
    #     loss_ = F.binary_cross_entropy(
    #         masked_infer, masked_gold, reduction='none')
    #     loss_ = loss_.view(batch_size)
    # else:
    #     loss = F.binary_cross_entropy(masked_infer, masked_gold)
    return loss
