import threading


def cookie_test():
    import requests
    civitai_cookie = "__Secure-civitai-token=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..o9E0O2vKpNnCWtB6.EOfQnHJr-gRewFTTErAboA38UymwyuDED35bicH12oBnFrFQ4_EMOZluxjMKi0g3Lm-mzOzzJbw0tBigTY8iAr6OGl81KPAAHIGtU-Y98eXT3P52cHFuo3Y-9FvwswxtsHCGtRm64MmTxGJKUivPBITpg7y89dzJ0YNp7OiPo8PN2qTVpHl2Jrl5Td591Z3ikMNIGHTEky8N_rX6QoGhVbDCrUW-MqcjUlE5enEMggUyYyCQxCYllMGUW4P1T4Ev5uXFRQWP221AHZI5gtOWQQMmaI98gYSL6x-q_ZnToa0UK-KQYIR_xIQFjvBrMk67ZFJLticbBs65q055teu2FbklMd38rrYe7NgHycrG2yaeQInIW20NBKaM0XGqu2IWCZAJpoUFR6ZKuwo-G_UDDFJJ7l09Rgk3Ww6_1g1KB88PpT_l1oN5OGyny_REG_eqWmNgeLO-vSP_ogwDiXxqgNCmlMfyXruK3lYO4ro1glHsgH2PwMUBQ2Rpjtao5rO7dzHdXOw5lVRbQvOk4ChuhmIQSXJRtQE8s55DNX6yQEPl0pLZsRewuQ-n5qJHOOJ4MB92q6i-6WukEoZich9qb1iwC4YhIOjBp0y55Ar51oUabQ9eNTruFxHFrflxEIjATNbq4WQMZfZi0Co4qM3Y-oEw6r_Xt56MYOUF10MjNazwfmoVr2VrXyLyWVhfQnJ7bebui4Qi21gZhwT6ryPxCdHOmghBonfbRKTpygYP5MuJ-uDNRC-477RRNure02SvC13aJa6OPJW55t-ABIz2LiUwSAqnuLGJSha6W3atFz2Ns8N61tzzba4jVUCErcZVYWjiN5MsVY9JU2VFnIeXRuV7BpEM2IKd7P_9lGhqcDHNiY0bON0IQ5K_xicdRUa_GEtOrgc-KQoErSfVi0YPArHGbw4FkHL1r_Ts_2LtNy35R9bJPb-gKEy6UOMQrM87_8ADJt31er-R4908LGjeFr2ZovgkL6lEQ7CFVL5QJLAmwGfACjN4u-gRwZbl_YbpsVCRfdPY0t4ZnZGupSH2fL64Tv7dumwjG92H9pZHtx0iteQxQbhsl7h_R_h5dgAo94bOEb3IyjUBqcP5jJ8ascE9SbYaTN-cQXhC-NOJzGHGfEhFYtKXXoCynOIUNkVNKDkGyf5knsLu5oW7QW9QjRafdaf-FblZLLGUyHELCSBAsm0VU3niTDB4S8KE96ou-B2jFHece5bOhVFrUtRyS7cjVh-gz7cfySBrbAciYTi-uMs.oiTi5nqQCd3bwiaeuzCGCA;"
    hf_cookie = "token=tNdvqKWNwfXlIhemmKYNbDXDgXRshHvwXOFUzXWWVZjkNbWRfszhynxgDXSmfjZDZZuBwywspRCUZURCZBgTuchTYyqxmayZtumKiNfXtxajLQVZGKgvLSDIsWIMIbbq"
    cookie = civitai_cookie + hf_cookie

    civitai_model = "https://civitai.com/api/download/models/450105?type=Model&format=SafeTensor&size=full&fp=fp16"
    hf_model = "https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/resolve/main/Juggernaut-XI-byRunDiffusion.safetensors?download=true"
    r = requests.get("https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true", headers={"cookie": cookie}, stream=True)
    print(r.headers.get("content-disposition"))
    print(r.url)

def test_pipe(pipe,prompt=None):
    if prompt is None:
        prompt = "instagram photo, front view, portrait photo of a 24 y.o woman, wearing dress, beautiful face, cinematic shot"
    args = [[25,0],[30,0],[35,0],[25,1.5],[30,1.5],[35,1.5],[25,3],[30,3],[35,3],[25,4.5],[30,4.5],[35,4.5],[25,6],[30,6],[35,6]]
    r = {}
    for arg in args:
        print(f"Step:{arg[0]},CFG:{arg[1]}")
        r.update({arg.__str__():pipe(prompt,num_inference_steps=arg[0],guidance_scale=arg[1]).images[0]})
    return

a = [1,2,None]
b = [item for item in a if item]

# from diffusers.models.model_loading_utils import load_state_dict
# from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
# from diffusers import FluxTransformer2DModel


def cat(tensors, dim=0):
    import torch
    device = tensors[0].device
    for t in tensors:
        if t.device != device:
            raise ValueError("The tensors are not on the same device.")

    dtype = tensors[0].dtype
    for t in tensors:
        if t.dtype != dtype:
            raise ValueError("The dtype of tensors are not the same.")

    # 计算拼接后的形状
    total_size = sum(tensor.size(dim) for tensor in tensors)
    size_list = [tensor.size() for tensor in tensors]

    # 创建一个新的空张量用于存储拼接后的结果
    result_size = list(size_list[0])  # 复制第一个张量的形状
    result_size[dim] = total_size  # 沿 dim 维度拼接后的大小
    result_tensor = torch.empty(result_size, dtype=dtype, device=device)

    # 使用切片手动拼接
    offset = 0
    for tensor in tensors:
        # 获取当前张量在拼接维度的大小
        tensor_size = tensor.size(dim)
        # 切片赋值
        result_tensor.narrow(dim, offset, tensor_size).copy_(tensor)
        offset += tensor_size

    return result_tensor


