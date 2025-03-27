from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Liên quan vụ việc CSGT bị tố đánh dân, trúng một cháu nhỏ đang ngủ, đang lan truyền trên mạng xã hội, Đại tá Nguyễn Văn Tảo, Phó Giám đốc Công an tỉnh Tiền Giang vừa có cuộc họp cùng Chỉ huy Công an huyện Châu Thành và một số đơn vị nghiệp vụ cấp tỉnh để chỉ đạo làm rõ thông tin."

ner_results = nlp(example)
print(ner_results)