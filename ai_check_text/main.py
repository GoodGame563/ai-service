from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("MRNH/mbart-russian-grammar-corrector")

tokenizer = MBart50TokenizerFast.from_pretrained("MRNH/mbart-russian-grammar-corrector", src_lang="ru_RU", tgt_lang="ru_RU")
input = tokenizer("I was here yesterday to studying",text_target="I was here yesterday to study", return_tensors='pt')

output = model.generate(input["input_ids"],attention_mask=input["attention_mask"],forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"])

print(output)