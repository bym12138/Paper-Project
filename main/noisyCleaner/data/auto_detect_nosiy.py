import re

class VersionNoiseDetector:
    def __init__(self):
        # 版本号正则表达式，匹配 1.0, 1.0.0, 2.0.3, 1.0.0-alpha 等
        self.version_pattern = r"\b\d+(\.\d+){1,2}(-\w+)?\b"
         # 定义 URL 正则表达式
        self.url_pattern = r'https?://[^\s]+'

    def is_noise_doc_change(self, message):
        # 判断消息长度是否足够
        if len(message.split()) < 5:
            return True  # 短句子可能缺乏具体信息

        # 检查是否包含版本号
        if re.search(self.version_pattern, message):
            return True  # 如果消息包含版本号，标记为噪音
        
        # 检查是否包含链接
        if re.search(self.url_pattern, message):
            return True  # 如果消息包含链接，标记为噪音

        # 检查是否包含具体的操作描述
        # if not self.has_specific_content(message):
        #     return True  # 如果没有具体的内容描述，标记为噪音

        return False


    def is_miscellaneous_noise(self, message):
        """
        判断 Miscellaneous 类型的消息是否为噪音：
        - 只包含版本号或版本号与"Release"的情况被视为噪音。
        - 包含完整句子描述的版本号不视为噪音。
        """
        message = message.strip().lower()

        # 如果消息只包含版本号，或者只有版本号加 "release"
        if re.fullmatch(self.version_pattern, message) or re.fullmatch(r"release\s+" + self.version_pattern, message):
            return True

        # 如果消息包含很少的词语，并且包含版本号 (可能是噪音)
        if len(message.split()) <= 3 and re.search(self.version_pattern, message):
            return True

        # 检查是否包含链接
        if re.search(self.url_pattern, message):
            return True  # 如果消息包含链接，标记为噪音

        # 否则，视为非噪音
        return False

    def detect_noise(self, message, message_type):
        """
        根据消息类型，检测消息是否为噪音。
        目前主要处理 Miscellaneous 类型，但可以扩展到其他类型。
        """
        # 检查是否为 Miscellaneous 类型
        if message_type.strip().lower() == "miscellaneous":
            if self.is_miscellaneous_noise(message):
                return "Noise"
            else:
                return "Non-Noise"
            
        # 检查是否为 Miscellaneous 类型
        elif message_type.strip().lower() == "documentation" or message_type.strip().lower() == "changes":
            if self.is_noise_doc_change(message):
                return "Noise"
            else:
                return "Non-Noise"
        
        # 其他类型的默认处理（目前认为非噪音）
        return "Non-Noise"


# 测试代码
detector = VersionNoiseDetector()

test_messages = [
    {"message": "1.4.0", "type": "Miscellaneous"},
    {"message": "Release 2.0.3", "type": "Miscellaneous"},
    {"message": "Release 2.0.3 with bug fixes", "type": "Miscellaneous"},
    {"message": "publish v1.3.7", "type": "Miscellaneous"},
    {"message": "Updated 1.4.0 for performance improvements", "type": "Miscellaneous"},
    {"message": "Updated 1.4.0 for performance improvements https://github.com/raml-org/raml-js-parser-2/issues/776", "type": "Miscellaneous"},
    {"message": "chore: bump up dependencies","type": "Documentation"},
    {"message": "chore: Use node 12","type": "Documentation"},
    {"message": "Release 2.0.3","type": "Documentation"},
    {"message": "docs: document utility functions","type": "Documentation"},
    {"message": "Allow tornado 5.x","type": "Documentation"},
    {"message": "apps/README.md document METIS graph output -G","type": "Documentation"},
    {"message": "apps/README.md document https://github.com/raml-org/raml-js-parser-2/issues/776","type": "Documentation"},
]

# 对每条消息进行测试
for test in test_messages:
    label = detector.detect_noise(test["message"], test["type"])
    print(f"Message: {test['message']}, Type: {test['type']}, Label: {label}")
