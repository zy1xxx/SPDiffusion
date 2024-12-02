import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    deprecate,
    replace_example_docstring,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers import StableDiffusionXLPipeline
import stanza
from nltk.tree import Tree, ParentedTree
import numpy as np
import random
import torch.nn.functional as F
from PIL import Image

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""
import copy
skip_nouns = ["photo", "bunches", "bunch", "front", "patch", "side",
              "pile", "piece"]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AttnProcessorStore(torch.nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self):
        super().__init__()
        self.attn_map=[]
        self.step=0
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        global write_flag,st_step_g
        
        if self.step>=20:
            self.step=0
        self.step=self.step+1
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if self.step<=st_step_g:
            if write_flag:
                self.attn_map.append(attention_probs)
            else:
                if len(self.attn_map)<self.step:
                    raise ValueError("attn_map is not enough")
                attention_probs=self.attn_map[self.step-1]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class SP_Attn(torch.nn.Module):
    def __init__(self, hidden_size = None, cross_attention_dim=None,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.attn_map=[]
        self.step=0
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        global SP_Mask_24,SP_Mask_48,height_g,width_g

        if hidden_states.shape[1] == (height_g//32) * (width_g//32):
            mask=SP_Mask_24
        else:
            mask=SP_Mask_48  
        
        if self.step>=20:
            self.step=0
        self.step=self.step+1
 
        if mask is not None:
            attention_mask = torch.zeros(mask.shape, dtype=hidden_states.dtype, device=hidden_states.device)
            attention_mask = attention_mask.masked_fill(mask == False, float('-inf'))
        else:
            attention_mask = None
        hidden_states = self.sp_attn_call(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        
        return hidden_states
        
    def sp_attn_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        global write_flag,st_step_g

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.step<=st_step_g:
            if write_flag:
                self.attn_map.append(attention_probs)
    
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

##functions for caption image
def concat_images_for_col(images):
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    new_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img
def concat_images_for_row(images):
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    new_img = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    return new_img

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class SPDiffusionPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler, image_encoder, feature_extractor, force_zeros_for_empty_prompt, add_watermarker) 
        self.set_SP_Attn()
        self.parser=stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)

    def get_pairs(self,text: str):
        nlp = self.parser
        parse_doc = nlp(text)
        tree = Tree.fromstring(str(parse_doc.sentences[0].constituency))
        tree = ParentedTree.convert(tree)

        def get_pairs(tree):
            pairs = []
            if type(tree) == str or tree is None:
                return []
            # Always suppose that the subject is in the last position of a concept
            # We here only consider the simplest case, taking no other labels like 'NNP', 'NNPS' into consideration
            if tree.label() in ['NN', 'NNS'] and tree.leaves()[0] == tree.parent().leaves()[-1]:
                cut_off = 0
                # if len(tree.parent().leaves()) == 2 and tree.parent()[0].label() == 'DT':
                #     cut_off = 1

                if tree.parent()[0].label() == 'DT':
                    cut_off = 1
                    
                concept = ' '.join(tree.parent().leaves()[cut_off:])
                # if len(tree.parent().leaves()[cut_off:]) == 1: return []

                if concept in skip_nouns: return []

                pairs = [{'subject': ' '.join(tree.leaves()), 
                        'attribute': ' '.join(concept.split(' ')[:-1]),
                        'concept': concept,
                        }]
            
            for subtree in tree:
                pairs += get_pairs(subtree)
            
            return pairs
        
        all_pairs = get_pairs(tree)

        all_concepts = [pair['concept'] for pair in all_pairs]
        # print(all_concepts)
        all_attributes = [pair['attribute'] for pair in all_pairs]
        # print(all_attributes)
        remove_list = []

        for p_id, concept in enumerate(all_concepts):
            for attribute in all_attributes:
                if concept in attribute:
                    remove_list.append(p_id)

        # print(remove_list)
        output = []
        for p_id, pair in enumerate(all_pairs):
            if p_id in remove_list:
                continue
            output.append(pair)
        return output

    def set_SP_Attn(self):
        attn_procs={}
        total_count=0
        self_total_count=0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if cross_attention_dim is not None and (name.startswith("up_blocks") or name.startswith("down_blocks") or name.startswith("mid_block")) :
                attn_procs[name] = SP_Attn()
                total_count +=1
            else:
                attn_procs[name] = AttnProcessorStore()
                self_total_count +=1
        self.unet.set_attn_processor(attn_procs)

    def clear_attn_map(self):
        total_count=0
        for name,attn in self.unet.attn_processors.items():
            if hasattr(attn, 'attn_map'):
                attn.attn_map=[]
            total_count +=1
    
    def aggregate_attn_map(self,is_cross=True):
        attn_map_store={"mid":[],"up":[],"down":[]}
        total_count=0
        for name,attn in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            #name
            if name.startswith("mid_block"):
                pos_name="mid"
            elif name.startswith("up_blocks"):
                pos_name="up"
            elif name.startswith("down_blocks"):
                pos_name="down"
            
            if attn.attn_map is None or attn.attn_map==[]:
                continue
            if is_cross:
                if cross_attention_dim is not None:
                    attn_map_store[pos_name].append(attn.attn_map)
                    total_count +=1
            else:
                if cross_attention_dim is None:
                    attn_map_store[pos_name].append(attn.attn_map)
                    total_count +=1
        return attn_map_store

    def clear_around(self,attn_map_col,attn_map_size=24):
        select=torch.zeros((attn_map_size,attn_map_size),dtype=torch.bool)
        select[0,:]=True
        select[-1,:]=True
        select[:,0]=True
        select[:,-1]=True
        attn_map_col[select.flatten()]=0
        return attn_map_col
    def filter_attn_map(self,attn_map_store,loc=['mid','down','up'],attn_map_size=24,step=-1):
        attn_map_ls=[]
        for key,attn_map_layers in attn_map_store.items():
            if key in loc:
                for attn_map_steps in attn_map_layers:
                    if attn_map_steps[0].shape[1]==attn_map_size*attn_map_size:
                        if step==-1:
                            attn_map_ls.extend(attn_map_steps)
                        elif type(step)!=int:
                            for s in step:
                                attn_map_ls.append(attn_map_steps[s])
                        else:                        
                            attn_map_ls.append(attn_map_steps[step])
                        
        avg_attn_map=torch.stack(attn_map_ls).mean(dim=0)
        
        avg_attn_map=avg_attn_map[avg_attn_map.shape[0]//2:]##condition attn map
        avg_attn_map=torch.mean(avg_attn_map,dim=0)
        return avg_attn_map
    
    def get_cross_mask(self,avg_attn_map_cross,concept_ls,cross_threshold,attn_map_size=24):
        attn_map_col_ls=[]
        for c_i_ls in concept_ls:
            attn_map_col=torch.mean(avg_attn_map_cross[:,c_i_ls],dim=-1)
            attn_map_col=(attn_map_col-attn_map_col.min())/(attn_map_col.max()-attn_map_col.min())#normalize
            attn_map_col=self.clear_around(attn_map_col,attn_map_size)
            attn_map_col=(attn_map_col-attn_map_col.min())/(attn_map_col.max()-attn_map_col.min())#normalize
            attn_map_col_ls.append(attn_map_col)
        image_mask_ls=[]
        for attn_map_col in attn_map_col_ls:
            img_mask = attn_map_col>cross_threshold
            image_mask_ls.append(img_mask)
            attn_map_col=attn_map_col.reshape((attn_map_size,attn_map_size))
        return image_mask_ls

    def get_self_mask(self,avg_attn_map_self,cross_mask_ls,self_threshold,attn_map_size=24):
        attn_map_col_ls=[]
        image_mask_ls=[]

        for image_mask in cross_mask_ls:
            attn_map_col=torch.mean(avg_attn_map_self[:,image_mask],dim=-1)
            attn_map_col=(attn_map_col-attn_map_col.min())/(attn_map_col.max()-attn_map_col.min())#normalize
            attn_map_col_ls.append(attn_map_col)

        fixed_attn_map_col_ls=[]
        for i,attn_map_col in enumerate(attn_map_col_ls):
            other_map_idx=list(range(len(attn_map_col_ls)))
            other_map_idx.remove(i)
            other_ls=[]
            for idx in other_map_idx:
                other_ls.append(attn_map_col_ls[idx])

            avg_other_attn_map=torch.mean(torch.stack(other_ls,dim=0),dim=0)
            attn_map_col=torch.relu((attn_map_col-avg_other_attn_map).float()).to(torch.float16)
            attn_map_col=(attn_map_col-attn_map_col.min())/(attn_map_col.max()-attn_map_col.min())
            fixed_attn_map_col_ls.append(attn_map_col)

        
        for attn_map_col in fixed_attn_map_col_ls:
            img_mask = attn_map_col>self_threshold
            image_mask_ls.append(img_mask)
            attn_map_col=attn_map_col.reshape((attn_map_size,attn_map_size))
        
        return image_mask_ls

    def gen_SP_Mask_(self,concept_bind_ls,SP_Mask_for_concept,attn_map_size=24):
        SP_Mask_24=torch.ones((attn_map_size*attn_map_size,77),dtype=torch.bool,device=self.device)
        for bind_ls,SP_Mask in zip(concept_bind_ls,SP_Mask_for_concept):        
            exception_ls=[x for x in concept_bind_ls if x != bind_ls]
            idx_ls=[item for sublist in exception_ls for item in sublist]
            token_select=torch.zeros((77,),dtype=torch.bool,device=self.device)
            token_select[idx_ls]=True
            bool_index_matrix = SP_Mask[:, None] & token_select[None, :]
            SP_Mask_24[bool_index_matrix] = False
        return SP_Mask_24
    
    def convert24maskTo48(self,SP_Mask_for_concept,attn_map_size=24):
        SP_Mask_48 = []
        for mask in SP_Mask_for_concept:
            mask=mask.reshape((attn_map_size,attn_map_size))
            mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, 24, 24]
            
            mask_resized = F.interpolate(mask, size=(attn_map_size*2, attn_map_size*2), mode='nearest')
        
            mask_resized = (mask_resized > 0.5).squeeze(0).squeeze(0)

            SP_Mask_48.append(mask_resized.flatten())
        return SP_Mask_48

    def get_ids(self,global_prompt,local_prompt):
        global_prompt_id_ls=self.tokenizer(global_prompt, return_tensors="pt")['input_ids'][0].tolist()
        local_prompt_id_ls=self.tokenizer(local_prompt, return_tensors="pt")['input_ids'][0].tolist()[1:-1]
        def find_sublist(big_list, sub_list):
            sub_len = len(sub_list)
            for i in range(len(big_list) - sub_len + 1):
                if big_list[i:i + sub_len] == sub_list:
                    return i, i + sub_len
            return -1, -1
        start_idx, end_idx = find_sublist(global_prompt_id_ls, local_prompt_id_ls)
        return list(range(start_idx, end_idx))

    def analyze_prompt(self, prompt):
        concept_bind_ls=[]
        concept_index_ls=[]
        # analyze prompt
        pairs=self.get_pairs(prompt)
        for pair in pairs:
            if pair['attribute']=="":
                continue
            concept_str=pair['concept']
            subject_str=pair['subject']
            concept_idx=self.get_ids(prompt,concept_str)
            subject_idx_loc=self.get_ids(concept_str,subject_str)
            subject_idx_glo=[concept_idx[0]+i-1 for i in subject_idx_loc]
            concept_bind_ls.append(concept_idx)
            concept_index_ls.append(subject_idx_glo)
        return concept_bind_ls,concept_index_ls

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 768,
        width: Optional[int] = 768,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        ##spd added
        st_step=2,
        filter_loc=['down','up','mid'],
        cross_threshold=0.9,
        self_threshold=0.1,
        run_sdxl=False,

        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        assert width==height
        assert num_images_per_prompt == 1
        if isinstance(prompt,list):
            assert len(prompt) == 1

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.scheduler2=copy.deepcopy(self.scheduler)
        # spd added 
        concept_bind_ls,concept_index_ls=self.analyze_prompt(prompt)
        
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # stage1
            global height_g,width_g,st_step_g
            height_g=height
            width_g=width
            st_step_g=st_step

            global SP_Mask_24,SP_Mask_48,write_flag
            SP_Mask_48=None
            SP_Mask_24=None
            write_flag=True
            latents1=latents

            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents1] * 2) if self.do_classifier_free_guidance else latents1

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents1.dtype
                latents1 = self.scheduler.step(noise_pred, t, latents1, **extra_step_kwargs, return_dict=False)[0]
                if latents1.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents1 = latents1.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents1 = callback_outputs.pop("latents", latents1)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents1)

                if not run_sdxl:
                    # early stop
                    if i>st_step:
                        break 
        
        

        # SP-Extraction
        attn_map_size=width//32
        mask_step=range(st_step)
        
        # cross attention map
        attn_map_store_cross=self.aggregate_attn_map(is_cross=True)
        avg_attn_map_cross=self.filter_attn_map(attn_map_store_cross,loc=filter_loc,attn_map_size=attn_map_size,step=mask_step)

        # self attention map
        attn_map_store_self=self.aggregate_attn_map(is_cross=False)
        avg_attn_map_self=self.filter_attn_map(attn_map_store_self,loc=filter_loc,attn_map_size=attn_map_size,step=mask_step)

        cross_mask_ls =self.get_cross_mask(avg_attn_map_cross,concept_index_ls,cross_threshold,attn_map_size=attn_map_size)
        SP_Mask_for_concept = self.get_self_mask(avg_attn_map_self, cross_mask_ls, self_threshold, attn_map_size=attn_map_size)

        SP_Mask_24=self.gen_SP_Mask_(concept_bind_ls,SP_Mask_for_concept,attn_map_size)
        SP_Mask_48=self.gen_SP_Mask_(concept_bind_ls,self.convert24maskTo48(SP_Mask_for_concept,attn_map_size),attn_map_size*2)
        SP_Mask_24=SP_Mask_24.to(device)
        SP_Mask_48=SP_Mask_48.to(device)

        write_flag=False
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # stage2
            latents2=latents
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents2] * 2) if self.do_classifier_free_guidance else latents2

                latent_model_input = self.scheduler2.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents2.dtype
                latents2 = self.scheduler2.step(noise_pred, t, latents2, **extra_step_kwargs, return_dict=False)[0]
                if latents2.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents2 = latents2.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents2 = callback_outputs.pop("latents", latents2)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler2.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler2, "order", 1)
                        callback(step_idx, t, latents2)

        self.clear_attn_map()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents1 = latents1.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                latents2 = latents2.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents1.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents1.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents1 = latents1 * latents_std / self.vae.config.scaling_factor + latents_mean
                latents2 = latents2 * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents1 = latents1 / self.vae.config.scaling_factor
                latents2 = latents2 / self.vae.config.scaling_factor

            if run_sdxl:
                image1 = self.vae.decode(latents1, return_dict=False)[0]
            
            image2 = self.vae.decode(latents2, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents2
        
        if not output_type == "latent":
            if run_sdxl:
                image1 = self.image_processor.postprocess(image1, output_type=output_type)
                image2 = self.image_processor.postprocess(image2, output_type=output_type)
                image=[concat_images_for_col(image1+image2)]
            else:
                image = self.image_processor.postprocess(image2, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

