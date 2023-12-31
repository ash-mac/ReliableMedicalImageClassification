import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from skimage.morphology import label, binary_erosion
from typing import List

class ClassDistinctivenessLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=1)
        self.device = device

    def view_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1)

    def forward(self, sal_tensor_list: List[torch.Tensor]) -> torch.Tensor:
        loss_list = torch.Tensor([]).to(self.device)
        for sal_comb in itertools.combinations(sal_tensor_list, 2):
            loss_list = torch.cat((loss_list, torch.unsqueeze(torch.abs(self.cossim(self.view_tensor(sal_comb[0]), self.view_tensor(sal_comb[1]))).mean(), dim=0)))
        return torch.mean(loss_list)

class SpatialCoherenceConv(nn.Module):
    def __init__(self, device, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device

    def kernels(self, image_channel):
        kernel1 = torch.zeros([image_channel, 1, self.kernel_size, self.kernel_size])
        kernel1[:, 0, self.kernel_size // 2, self.kernel_size // 2] = 1
        kernel2 = torch.ones([image_channel, 1, self.kernel_size, self.kernel_size]) * -2
        kernel2[:, 0, self.kernel_size // 2, self.kernel_size // 2] = self.kernel_size * self.kernel_size - 2
        kernel3 = torch.ones([image_channel, 1, self.kernel_size, self.kernel_size])
        return kernel1.to(self.device), kernel2.to(self.device), kernel3.to(self.device)

    @staticmethod
    def image_for_conv(image):
        image_sq = torch.square(image)
        return image, image_sq

    def multi_convs(self, image):
        # this part is modified
        if image.shape[1]<1:
            return 0
        # ------
        k1, k2, k3 = self.kernels(image.shape[1])
        im, im_sq = self.image_for_conv(image)
        k1.requires_grad = False
        k2.requires_grad = False
        k3.requires_grad = False
        conv1 = F.conv2d(im, k1, padding='same', groups=image.shape[1])
        conv2 = F.conv2d(im, k2, padding='same', groups=image.shape[1])
        conv3 = F.conv2d(im_sq, k3, padding='same', groups=image.shape[1])

        # print(conv1.shape)
        convsum = conv1 * conv2 + conv3
        #print(convsum)
        return torch.mean(torch.sum(convsum, dim=(1, 2, 3)))

    @staticmethod
    def threshold_otsu_numpy(image=None, nbins=256):
        # Check if the image has more than one intensity value; if not, return that
        # value
        if image is not None:
            first_pixel = image.ravel()[0]
            if np.all(image == first_pixel):
                return first_pixel

        counts, bin_edges = np.histogram(image, bins=nbins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        # class probabilities for all possible thresholds
        weight1 = np.cumsum(counts)
        weight2 = np.cumsum(counts[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(counts * bin_centers) / weight1
        mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of ``weight1``/``mean1`` should pair with zero values in
        # ``weight2``/``mean2``, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[idx]

        return threshold

    @staticmethod
    def mask(self, sal_map_tensor, process=binary_erosion):
        sal_map_numpy = sal_map_tensor.detach().cpu().numpy()
        threshold = self.threshold_otsu_numpy(sal_map_numpy)
        binary_mask = (sal_map_numpy > threshold).astype(np.int32)
        processed_mask = process(binary_mask)
        labels = label(processed_mask, background=0)
        label_stack = np.zeros([1, np.max(labels)] + list(labels.shape[-2:]))
        for i in range(1, np.max(labels)):
            label_stack[0, i - 1] = labels == i
        return torch.from_numpy(label_stack.astype(np.float32)).to(self.device)

    def forward(self, sal_tensor_list, device):
        """
        :param sal_tensor_list: a list of saliency tensors calculated for each class
        :return:
        """

        sum_sdiff = torch.zeros(1).to(device)
        for sal_tensor in sal_tensor_list:
            for batch_index in range(sal_tensor.shape[0]):
                label_stack = self.mask(self, sal_tensor[batch_index])
                masked_sal_tensor = label_stack * sal_tensor[batch_index:batch_index + 1]
                sum = self.multi_convs(masked_sal_tensor)
                #print(sum)
                sum_sdiff += sum
        return sum_sdiff[0]
        #return sum_sdiff[0]/(len(sal_tensor_list)*sal_tensor_list[0].shape[0])

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss