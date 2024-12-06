import json
from pydantic import BaseModel



class description_words(BaseModel):
    description: str
    words: list[str]

main_element = description_words(description="",words=[])
elements = []

with open("response.json", "r", encoding='utf-8') as write_file:
    t = list(json.load(write_file))
    words = []
    
    first_element = t.pop()
    main_element.description = first_element["description"]
    main_element.words = [e["raw_keyword"] for e in first_element["seo"]]
    for element in t:
        elements.append(description_words(description=element["description"], words=[e["raw_keyword"] for e in element["seo"]]))

print(len(elements))


