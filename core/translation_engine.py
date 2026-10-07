"""Local HY-MT inference. Only model download requires network access."""
import gc


class TranslationEngine:
    def __init__(self, model_path="tencent/HY-MT1.5-1.8B", device="auto",
                 local_files_only=False, target_language="Spanish", max_new_tokens=512):
        self.model_path = model_path
        self.device = device
        self.local_files_only = local_files_only
        self.target_language = target_language
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.cache = {}

    def initialize(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. Selecciona cpu o auto.")
        dtype = torch.float32 if device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        options = dict(local_files_only=self.local_files_only, trust_remote_code=False)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **options)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=dtype, **options).to(device).eval()
        except Exception as e:
            if self.local_files_only:
                raise RuntimeError(
                    f"Modelo '{self.model_path}' no encontrado localmente. "
                    f"Desactiva 'translation_offline' para descargarlo."
                ) from e
            raise

    def is_chinese(self, text):
        """Detecta si el texto contiene caracteres chinos."""
        if not text:
            return False
        # Rango de caracteres chinos: CJK Unified Ideographs
        return any('\u3400' <= char <= '\u9fff' for char in text)

    def translate(self, text):
        text = text.strip()
        if not text:
            return ""
        if text in self.cache:
            return self.cache[text]
        self.initialize()
        import torch
        # Detectar si el texto está en chino para usar el prompt apropiado
        if self.is_chinese(text):
            prompt = f"将以下文本翻译为{self.target_language}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        else:
            prompt = (f"Translate the following segment into {self.target_language}, "
                      f"without additional explanation.\n\n{text}")
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=False, return_tensors="pt").to(self.model.device)
        context_limit = getattr(self.model.config, "max_position_embeddings", 8192)
        if tokens.shape[-1] + self.max_new_tokens > context_limit:
            raise ValueError("Segmento demasiado largo para HY-MT; divide el texto.")
        with torch.inference_mode():
            output = self.model.generate(
                tokens, attention_mask=torch.ones_like(tokens),
                max_new_tokens=self.max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, tokens.shape[-1]:]
        if generated.numel() >= self.max_new_tokens and generated[-1].item() != self.tokenizer.eos_token_id:
            raise RuntimeError("HY-MT alcanzó el límite de salida; aumenta translation_max_new_tokens.")
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not result:
            raise RuntimeError("HY-MT devolvió una traducción vacía.")
        self.cache[text] = result
        return result

    def close(self):
        self.model = None
        self.tokenizer = None
        self.cache.clear()
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
