import torch

def get_device_info(gpu_id: int):
    if torch.cuda.is_available():
        try:
            device = torch.device(f'cuda:{gpu_id}')
            gpu_name = torch.cuda.get_device_name(gpu_id)
        except Exception:
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        device = torch.device('cpu')
        gpu_name = 'cpu'
    return device, gpu_name

