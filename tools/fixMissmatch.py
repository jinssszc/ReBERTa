import torch
ckpt = torch.load("models/saved/model.pt", map_location="cpu", weights_only=False)
ckpt["config"]["window_size"] = ckpt["model_state_dict"]["first_encoder.embeddings.position_embeddings.weight"].shape[0]
torch.save(ckpt, "models/saved/model.pt")