import torch
import torchvision

# An instance of your model.
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
torch.save(model.state_dict(), "model_1.pt")

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 384, 384)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("model.pt")