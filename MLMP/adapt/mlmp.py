import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import peft
import logging

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'




class MLMP:
    """
    Multi-Layer Multi-Prompt adaptation for open-vocabulary semantic segmentation (OVSS) models.

    Performs test-time adaptation by updating only the visual encoder LayerNorm parameters via 
    multiple prompt and multiple level optimization. 
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, vision_outputs=(-1,), 
                 alpha_cls=0.0, steps=10, prompt_dir='prompts.yaml', 
                 prompt_integration='loss', runtime_calculation=False,
                 device='cpu',
                 ):
        """
        Initialize the MLMP adaptation module.

        Args:
            ovss_type (str): Identifier for the open-vocabulary segmentation model to load.
            ovss_backbone (str): Name of the backbone architecture within the OVSS model.
            lr (float): Learning rate for the LayerNorm optimizer.
            classes (List[str]): List of class names used to generate text embeddings.
            alpha_cls (float, optional): Weight for the classification‐entropy term. Defaults to 0.0.
            steps (int, optional): Number of test-time adaptation iterations per sample. Defaults to 10.
            prompt_dir (str, optional): Path to the YAML file containing prompt templates. Defaults to 'prompts.yaml'.
            prompt_integration (str, optional): Integration mode for prompts, either 'loss' or 'text'. Defaults to 'loss'.
            runtime_calculation (bool, optional): Whether to track adaptation/evaluation runtimes. Defaults to False.
            device (str, optional): Compute device, e.g., 'cpu' or 'cuda'. Defaults to 'cpu'.
        """

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        self.vision_outputs = vision_outputs
        print(f"+++ The output layers from vision encoder that will be used: {self.vision_outputs}")

        self.alpha_cls = alpha_cls
        self.steps = steps
        self.prompt_dir = prompt_dir
        self.runtime = runtime_calculation
        self.device = device
        # self.device = "cpu"

        # import ipdb; ipdb.set_trace()


        # Load the OVSS model and tokenizer
        self.model, self.tokenize = load_ovss(self.ovss_type, self.ovss_backbone, device=self.device)

        # import ipdb; ipdb.set_trace()
        # for name, param in  list(self.model.named_parameters()):
        #     param.requires_grad = False
        if self.prompt_dir:
            # Load the prompt templates
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            # print the number of prompt templates
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        
        assert prompt_integration in ['loss', 'text'], "prompt_integration should be either on 'loss' or 'text'"
        self.prompt_integration = prompt_integration

        # Set the gradients for LayerNorm layers only for visual encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        self.model.visual = self.set_ln_grads(self.model.visual)
        #self.model.add_lora()


        # for name, params in self.model.state_dict():
        #     # if "transformer" in name:
        #     #     if clip_finetune == "prompt":
        #     #         params.requires_grad = True if "prompt" in name else False
        #     #     elif clip_finetune == "attention":
        #     #         if "attn" in name:
        #     #             # QV fine-tuning for attention blocks
        #     #             params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
        #     #         elif "position" in name:
        #     #             params.requires_grad = True
        #     #         else:
        #     #             params.requires_grad = False
        #     #     elif clip_finetune == "full":
        #     #         params.requires_grad = True
        #     #     else:
        #     #         params.requires_grad = False
        #     # else:
        #     #     params.requires_grad = False
        #     params.requires_grad = False


        # Collect the LayerNorm parameters
        params, _ = self.collect_ln_params(self.model.visual)

        # print the parameters
        print_clip_parameters(self.model)

        # Set the optimizer
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # print the parameters passed to the optimizer
        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        # extracting text features
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes,  self.prompt_templates, average=True).squeeze() # (class, 512)

        # define variables to store adaptation and evaluation duration
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, x):
        """
        Forward pass with adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            List[float]: Loss values recorded at each adaptation iteration.
        """

        self.reset()
        loss_report = self.perform_adaptation(x)
        return loss_report


    @torch.no_grad() 
    def evaluate(self, x):
        """
        Forward pass without adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).

        """

        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x[-1], True, vision_outputs=self.vision_outputs, 
                                  interpolate=True, vision_out_type="adaptive_weighted_mean", 
                                  save_weights=True) # (#template, batch_size, #classes, H, W)
        logits = logits[0]
        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2-t1)

        return logits


    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)




    
    def add_lora(self):

        model = self.model
        logger = logging.getLogger("detectron2.trainer")

        MODUELS = [
    'visual.transformer.resblocks.0.attn', 
    'visual.transformer.resblocks.0.mlp.c_fc', 
    'visual.transformer.resblocks.0.mlp.c_proj', 
    'visual.transformer.resblocks.1.attn', 
    'visual.transformer.resblocks.1.mlp.c_fc', 
    'visual.transformer.resblocks.1.mlp.c_proj', 
    'visual.transformer.resblocks.2.attn', 
    'visual.transformer.resblocks.2.mlp.c_fc', 
    'visual.transformer.resblocks.2.mlp.c_proj', 
    'visual.transformer.resblocks.3.attn', 
    'visual.transformer.resblocks.3.mlp.c_fc', 
    'visual.transformer.resblocks.3.mlp.c_proj', 
    'visual.transformer.resblocks.4.attn', 
    'visual.transformer.resblocks.4.mlp.c_fc', 
    'visual.transformer.resblocks.4.mlp.c_proj', 
    'visual.transformer.resblocks.5.attn', 
    'visual.transformer.resblocks.5.mlp.c_fc', 
    'visual.transformer.resblocks.5.mlp.c_proj', 
    'visual.transformer.resblocks.6.attn', 
    'visual.transformer.resblocks.6.mlp.c_fc', 
    'visual.transformer.resblocks.6.mlp.c_proj', 
    'visual.transformer.resblocks.7.attn', 
    'visual.transformer.resblocks.7.mlp.c_fc', 
    'visual.transformer.resblocks.7.mlp.c_proj', 
    'visual.transformer.resblocks.8.attn', 
    'visual.transformer.resblocks.8.mlp.c_fc', 
    'visual.transformer.resblocks.8.mlp.c_proj', 
    'visual.transformer.resblocks.9.attn', 
    'visual.transformer.resblocks.9.mlp.c_fc', 
    'visual.transformer.resblocks.9.mlp.c_proj', 
    'visual.transformer.resblocks.10.attn', 
    'visual.transformer.resblocks.10.mlp.c_fc', 
    'visual.transformer.resblocks.10.mlp.c_proj', 
    'visual.transformer.resblocks.11.attn', 
    'visual.transformer.resblocks.11.mlp.c_fc', 
    'visual.transformer.resblocks.11.mlp.c_proj', 
    'visual.transformer.resblocks.12.attn', 
    'visual.transformer.resblocks.12.mlp.c_fc', 
    'visual.transformer.resblocks.12.mlp.c_proj', 
    'visual.transformer.resblocks.13.attn', 
    'visual.transformer.resblocks.13.mlp.c_fc', 
    'visual.transformer.resblocks.13.mlp.c_proj', 
    'visual.transformer.resblocks.14.attn', 
    'visual.transformer.resblocks.14.mlp.c_fc', 
    'visual.transformer.resblocks.14.mlp.c_proj', 
    'visual.transformer.resblocks.15.attn', 
    'visual.transformer.resblocks.15.mlp.c_fc', 
    'visual.transformer.resblocks.15.mlp.c_proj', 
    'visual.transformer.resblocks.16.attn', 
    'visual.transformer.resblocks.16.mlp.c_fc', 
    'visual.transformer.resblocks.16.mlp.c_proj', 
    'visual.transformer.resblocks.17.attn', 
    'visual.transformer.resblocks.17.mlp.c_fc', 
    'visual.transformer.resblocks.17.mlp.c_proj', 
    'visual.transformer.resblocks.18.attn', 
    'visual.transformer.resblocks.18.mlp.c_fc', 
    'visual.transformer.resblocks.18.mlp.c_proj', 
    'visual.transformer.resblocks.19.attn', 
    'visual.transformer.resblocks.19.mlp.c_fc', 
    'visual.transformer.resblocks.19.mlp.c_proj', 
    'visual.transformer.resblocks.20.attn', 
    'visual.transformer.resblocks.20.mlp.c_fc', 
    'visual.transformer.resblocks.20.mlp.c_proj', 
    'visual.transformer.resblocks.21.attn', 
    'visual.transformer.resblocks.21.mlp.c_fc', 
    'visual.transformer.resblocks.21.mlp.c_proj', 
    'visual.transformer.resblocks.22.attn', 
    'visual.transformer.resblocks.22.mlp.c_fc', 
    'visual.transformer.resblocks.22.mlp.c_proj', 
    'visual.transformer.resblocks.23.attn', 
    'visual.transformer.resblocks.23.mlp.c_fc', 
    'visual.transformer.resblocks.23.mlp.c_proj', 
    'transformer.resblocks.0.attn', 
    'transformer.resblocks.0.mlp.c_fc', 
    'transformer.resblocks.0.mlp.c_proj', 
    'transformer.resblocks.1.attn', 
    'transformer.resblocks.1.mlp.c_fc', 
    'transformer.resblocks.1.mlp.c_proj', 
    'transformer.resblocks.2.attn', 
    'transformer.resblocks.2.mlp.c_fc', 
    'transformer.resblocks.2.mlp.c_proj', 
    'transformer.resblocks.3.attn', 
    'transformer.resblocks.3.mlp.c_fc', 
    'transformer.resblocks.3.mlp.c_proj', 
    'transformer.resblocks.4.attn', 
    'transformer.resblocks.4.mlp.c_fc', 
    'transformer.resblocks.4.mlp.c_proj', 
    'transformer.resblocks.5.attn', 
    'transformer.resblocks.5.mlp.c_fc', 
    'transformer.resblocks.5.mlp.c_proj', 
    'transformer.resblocks.6.attn', 
    'transformer.resblocks.6.mlp.c_fc', 
    'transformer.resblocks.6.mlp.c_proj', 
    'transformer.resblocks.7.attn', 
    'transformer.resblocks.7.mlp.c_fc', 
    'transformer.resblocks.7.mlp.c_proj', 
    'transformer.resblocks.8.attn', 
    'transformer.resblocks.8.mlp.c_fc', 
    'transformer.resblocks.8.mlp.c_proj', 
    'transformer.resblocks.9.attn', 
    'transformer.resblocks.9.mlp.c_fc', 
    'transformer.resblocks.9.mlp.c_proj', 
    'transformer.resblocks.10.attn', 
    'transformer.resblocks.10.mlp.c_fc', 
    'transformer.resblocks.10.mlp.c_proj', 
    'transformer.resblocks.11.attn', 
    'transformer.resblocks.11.mlp.c_fc', 
    'transformer.resblocks.11.mlp.c_proj', 
    ]

        config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=MODUELS,
            bias="none",
            use_rslora=False,
            use_dora=False,
        )
        peft_model = peft.get_peft_model(model, config, adapter_name="cityscapes")
        
        logger.info("LoRAs injected for training.")

        return peft_model



    def load_lora(model): # Loads the lora
        logger = logging.getLogger("detectron2.trainer")
        model = peft.PeftModel.from_pretrained(
            model, cfg.MODEL.LORA.DB_PATH + cfg.MODEL.LORA.NAME, cfg.MODEL.LORA.NAME
        ) # DB_PATH where loras are saved (parameters)
        # model wrapped -> trained Lora
        logger.info("Pre-trained LoRA loaded.")
        return




    def perform_adaptation(self, x):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
        
        Returns:
            List[float]: Recorded loss values for each adaptation iteration.
        """

        t1 = time.time()
        loss_report = []
        for iter in range(self.steps):
            if self.prompt_integration == 'loss':
                logits, _, _, cls_logits = self.model(x, self.text_x[:-1], True, interpolate=False,
                                                      vision_outputs=self.vision_outputs, return_vanilla_cls=True, 
                                                      vision_out_type="mean") # (#templates, batch_size, #class, W, H)
                
                # adapt
                entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (#template, batch_size, H, W)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                
                # Average over all prompt templates, pixels and batch samples
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()


            elif self.prompt_integration == 'text':
                logits, _, _ = self.model(x, self.text_x[-1], True, interpolate=False,
                                         vision_outputs=self.vision_outputs) # (1, batch_size, #classes, H, W)                                         
                entropy_per_pixel = self.softmax_entropy(logits)  # Shape: (batch_size, H, W)

                # Average over all pixels and batch samples
                loss = entropy_per_pixel.mean()

            else:
                raise Exception("prompt_integration should be either on 'loss' or 'text'")
        
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        t2 = time.time()
        if self.runtime:
            self.adapt_times.append(t2-t1)

        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        """
        Extracts text embeddings for given class names and prompts.

        Args:
            class_names: List of class names to generate text embeddings for.
            prompts: List of prompt templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different prompt templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and prompts.
        """
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)  # Shape: (#templates, 512)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)  # Shape: (512,)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                # add the averaged embeddings to the original embeddings
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

        Args:
            model: The model whose LayerNorm layers' gradients are to be set.

        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                # m.requires_grad_(False)
            # if isinstance(m, nn.Linear):
            #     m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.

        Args:
            model: The model from which to collect LayerNorm parameters.

        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # import ipdb; ipdb.set_trace()
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
            # if isinstance(m, nn.ModuleDict):
            # if isinstance(m, nn.Linear):
            #     # import ipdb; ipdb.set_trace()
            #     for np, p in m.named_parameters():
            #         if "cityscapes" in nm:
            #             if np in ['weight', 'bias']:
            #                 params.append(p)
            #                 names.append(f"visual.{nm}.{np}")
            print(names)
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """

        # import ipdb; ipdb.set_trace()
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        # import ipdb; ipdb.set_trace()

        # for k in list(checkpoint["model"].keys()):
        #     #new_k = "base_model.model." + k
        #     # new_k = k.replace("base_model.model.module.sem_seg_head", "sem_seg_head")
        #     new_k = k.replace(".base_layer.", ".")
        #     #new_k = k.replace("sem_seg_head.", "base_model.model.module.sem_seg_head.")
        #     checkpoint["model"][new_k]  = checkpoint["model"].pop(k)


        # for k in list(model_state.keys()):
        #     if "c_fc" in k:
        #         new_k = k.replace("c_fc", "c_fc.base_layer")
        #         model_state[new_k]  = model_state.pop(k)
        #     if "c_proj" in k:
        #         new_k = k.replace("c_proj", "c_proj.base_layer")
        #         model_state[new_k]  = model_state.pop(k)
        #     if "attn" in k:
        #         new_k = k.replace("attn", "attn.base_layer")
        #         model_state[new_k]  = model_state.pop(k)  visual.transformer.resblocks.0.attn.base_layer.in_proj_bias


        # import ipdb; ipdb.set_trace()


        # for k in list(model_state.keys()):
        #     if "c_fc" in k:
        #         new_k = k.replace("c_fc", "c_fc.base_layer")
        #         model_state[new_k]  = model_state.pop(k)
        #     if "c_proj" in k:
        #         cproj_k = k.replace("c_proj", "c_proj.base_layer")
        #         model_state[cproj_k]  = model_state.pop(k)
        #     if "attn" in k:
        #         attn_k = k.replace("attn", "attn.base_layer")
        #         model_state[attn_k]  = model_state.pop(k)

        model.load_state_dict(model_state, strict=True)
        # import ipdb; ipdb.set_trace()
        # model.load_state_dict(model_state, strict=False)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

