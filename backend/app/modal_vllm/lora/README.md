# LoRA adapters

One folder per adapter, each holding the files a vLLM/PEFT LoRA adapter has
(`adapter_config.json`, `adapter_model.safetensors`, ...):

```
lora/
  grounding/    -> /loras/grounding
  vqa/          -> /loras/vqa
  captioning/   -> /loras/captioning
  sar_ir/       -> /loras/sar_ir
```

`modal deploy` copies this folder into the container at `/loras`, so redeploy after changing an adapter.
The backend then needs the in-container paths:

```bash
export LORA_GROUNDING_PATH=/loras/grounding
export LORA_VQA_PATH=/loras/vqa
export LORA_CAPTIONING_PATH=/loras/captioning
export LORA_SAR_IR_PATH=/loras/sar_ir
```
