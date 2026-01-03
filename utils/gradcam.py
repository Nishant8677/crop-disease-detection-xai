import torch
import numpy as np
import cv2

def generate_gradcam(model, input_tensor, target_layer):
    model.eval()
    torch.set_grad_enabled(True)

    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_class].backward()

    gradients = activations.grad
    handle.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)

    cam = torch.relu(cam)
    cam = cam.squeeze()
    cam -= cam.min()
    cam /= cam.max()

    return cam.detach().cpu().numpy(), pred_class


def overlay_gradcam(image, cam, alpha=0.45):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
