"""SD1.4 TINA+ inversion, retaining the paper implementation's updates."""
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler


class TINA:
    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                       beta_schedule="scaled_linear", clip_sample=False,
                                       set_alpha_to_one=False)
        self.scheduler.set_timesteps(args.tina_num_ddim_steps)

    @torch.no_grad()
    def noise_prediction(self, latent, timestep, context):
        # Historical SD implementation: do not differentiate through UNet.
        return self.task.target_unet_sd(latent, timestep, encoder_hidden_states=context).sample

    @torch.no_grad()
    def next_step(self, model_output, timestep, sample):
        previous = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha = self.scheduler.alphas_cumprod[previous] if previous >= 0 else self.scheduler.final_alpha_cumprod
        next_alpha = self.scheduler.alphas_cumprod[timestep]
        predicted_clean = (sample - (1 - alpha) ** 0.5 * model_output) / alpha ** 0.5
        return next_alpha ** 0.5 * predicted_clean + (1 - next_alpha) ** 0.5 * model_output

    def energy_loss(self, latent, clean, timestep):
        z = latent.float()
        z0 = clean.detach().float()
        alpha = self.scheduler.alphas_cumprod[int(timestep)].to(z.device, z.dtype)
        variance_noise = (1 - alpha).clamp_min(0)
        dimensions = float(z.numel())
        clean_energy = z0.square().sum()
        expected = alpha * clean_energy + variance_noise * dimensions
        variance = 2 * variance_noise.square() * dimensions + 4 * alpha * variance_noise * clean_energy
        energy_score = (z.square().sum() - expected) / variance.clamp_min(1e-12).sqrt()
        return torch.relu(-energy_score - self.args.tina_energy_tau).square()

    def invert(self, image):
        args = self.args
        task = self.task
        with torch.no_grad():
            clean = task.vae.encode(image.unsqueeze(0).to(task.device)).latent_dist.mean * 0.18215
            ids = task.tokenizer([""], padding="max_length",
                                 max_length=task.tokenizer.model_max_length, return_tensors="pt").input_ids
            # Original CLIP forward for inversion; wrapper for LMS generation.
            context = task.text_encoder(ids.to(task.device))[0]
        latent = clean.clone().detach()
        n = args.tina_num_ddim_steps
        k = int(round(args.tina_fmi_ratio * n))
        if k:
            timestep = self.scheduler.timesteps[n - k]
            generator = torch.Generator(device=task.device).manual_seed(args.tina_fmi_seed)
            noise = torch.randn(clean.shape, generator=generator, device=task.device, dtype=clean.dtype)
            alpha = self.scheduler.alphas_cumprod[int(timestep)].to(clean.device, clean.dtype)
            latent = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        for i in range(k, n):
            timestep = self.scheduler.timesteps[n - i - 1]
            previous = latent.clone().detach()
            estimate = self.next_step(self.noise_prediction(latent, timestep, context), timestep, previous)
            optimal = estimate.clone().detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([optimal], lr=args.tina_lr)
            for _ in range(args.tina_opt_round):
                optimizer.zero_grad()
                noise = self.noise_prediction(optimal, timestep, context)
                predicted = self.next_step(noise, timestep, previous)
                loss = F.mse_loss(optimal, predicted)
                normalized_t = float(timestep) / self.scheduler.config.num_train_timesteps
                if max(args.tina_energy_t_min, args.tina_fmi_ratio) <= normalized_t <= args.tina_energy_t_max:
                    loss = loss + args.tina_energy_weight * self.energy_loss(optimal, clean, timestep)
                loss.backward()
                optimizer.step()
            latent = optimal.clone().detach()
        return latent, task.decode(clean, rounded=False)
