# TRELLIS

🚀 **[Try TRELLIS on Replicate](https://replicate.com/firtoz/trellis)** - Generate 3D models from images in your browser!

> **Note**: This Replicate deployment is maintained by firtoz, a fan of the TRELLIS project, and is not officially affiliated with Microsoft or the TRELLIS team. All rights, licenses, and intellectual property belong to Microsoft. For the original project, please visit [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS).

TRELLIS is a powerful 3D asset generation model that converts text or image prompts into high-quality 3D assets. This Replicate deployment focuses on the image-to-3D generation capabilities of TRELLIS.

<img src="https://github.com/microsoft/TRELLIS/blob/main/assets/teaser.png?raw=true" width="100%">

<!-- Updates -->
## ⏩ Updates
**12/18/2024**
- Implementation of multi-image conditioning for TRELLIS-image model. ([#7](https://github.com/microsoft/TRELLIS/issues/7)). This is based on tuning-free algorithm without training a specialized model, so it may not give the best results for all input images.
- Add Gaussian export in `app.py` and `example.py`. ([#40](https://github.com/microsoft/TRELLIS/issues/40))

<!-- TODO List -->
## 🚧 TODO List
- [x] Release inference code and TRELLIS-image-large model
- [ ] Release TRELLIS-text model series
- [ ] Release training code and data

## Model Description

TRELLIS uses a unified Structured LATent (SLAT) representation that enables generation of different 3D output formats. The model deployed here is TRELLIS-image-large, which contains 1.2B parameters and is trained on a diverse dataset of 500K 3D objects.

Key features:
- Generate high-quality 3D assets from input images
- Multiple output formats: 3D Gaussians, Radiance Fields, and textured meshes
- Detailed shape and texture generation
- Support for various viewpoint renderings

For more examples and to try it directly in your browser, visit the [Replicate model page](https://replicate.com/firtoz/trellis).

## Input Format

The model accepts:
- An input image (PNG or JPEG format)
- Optional parameters for controlling the generation process

## Output Format

The model outputs:
- A GLB file containing the generated 3D model with textures
- Preview renders from multiple angles
- Optional: Raw 3D Gaussians or Radiance Field representations

## Example Usage

```python
import replicate

output = replicate.run(
    "firtoz/trellis:version",
    input={
        "seed": 0,
        "image": "https://replicate.delivery/pbxt/M6rvlcKpjcTijzvLfJw8SCWQ74M1jrxowbVDT6nNTxREcvxO/ephemeros_cartoonish_character_art_cyberpunk_crocodile_white_ba_486fb649-bc68-46a0-b429-751b43734b89.png",
        "texture_size": 1024,
        "mesh_simplify": 0.95,
        "generate_color": True,
        "generate_model": True,
        "randomize_seed": True,
        "generate_normal": True,
        "ss_sampling_steps": 12,
        "slat_sampling_steps": 12,
        "ss_guidance_strength": 7.5,
        "slat_guidance_strength": 3
    }
)
print(output)
```

### Local Usage

```python
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open("assets/example_image/T.png")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave("sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("sample.ply")
```

After running the code, you will get the following files:
- `sample_gs.mp4`: a video showing the 3D Gaussian representation
- `sample_rf.mp4`: a video showing the Radiance Field representation
- `sample_mesh.mp4`: a video showing the mesh representation
- `sample.glb`: a GLB file containing the extracted textured mesh
- `sample.ply`: a PLY file containing the 3D Gaussian representation

### Web Demo

[app.py](app.py) provides a simple web demo for 3D asset generation. Since this demo is based on [Gradio](https://gradio.app/), additional dependencies are required:
```sh
. ./setup.sh --demo
```

After installing the dependencies, you can run the demo with the following command:
```sh
python app.py
```

Then, you can access the demo at the address shown in the terminal.

***The web demo is also available on [Hugging Face Spaces](https://huggingface.co/spaces/JeffreyXiang/TRELLIS)!***

<!-- License -->
## ⚖️ License

TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE). The following submodules may have different licenses:
- [**diffoctreerast**](https://github.com/JeffreyXiang/diffoctreerast): We developed a CUDA-based real-time differentiable octree renderer for rendering radiance fields as part of this project. This renderer is derived from the [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) project and is available under the [LICENSE](https://github.com/JeffreyXiang/diffoctreerast/blob/master/LICENSE).

- [**Modified Flexicubes**](https://github.com/MaxtirError/FlexiCubes): In this project, we used a modified version of [Flexicubes](https://github.com/nv-tlabs/FlexiCubes) to support vertex attributes. This modified version is licensed under the [LICENSE](https://github.com/nv-tlabs/FlexiCubes/blob/main/LICENSE.txt).

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```

## Links

- [Project Page](https://trellis3d.github.io)
- [Paper](https://arxiv.org/abs/2412.01506)
- [GitHub Repository](https://github.com/microsoft/TRELLIS)
- [Hugging Face Demo](https://huggingface.co/spaces/JeffreyXiang/TRELLIS)