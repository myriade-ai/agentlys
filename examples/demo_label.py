from agentlys import Agentlys


def label_item(category: str):
    # TODO: Implement function
    raise NotImplementedError()


classifierGPT = Agentlys(instruction="you classify title")
classifierGPT.add_function(label_item)

text = "The new iPhone is out"
for message in classifierGPT.run_conversation(text):
    print(message.to_markdown())
