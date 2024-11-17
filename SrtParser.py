import re
from dataclasses import dataclass
from typing import List, Tuple
from openai import OpenAI
import json


@dataclass
class SubtitleEntry:
    index: int
    timestamp: str
    content: str

    def __str__(self):
        return f"{self.index}\n{self.timestamp}\n{self.content}\n"


class SrtParser:
    @staticmethod
    def parse(file_path: str) -> List[SubtitleEntry]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分割每个字幕块
        subtitle_blocks = re.split(r'\n\n+', content.strip())
        entries = []

        for block in subtitle_blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                index = int(lines[0])
                timestamp = lines[1]
                content = '\n'.join(lines[2:])
                entries.append(SubtitleEntry(index, timestamp, content))

        return entries


class SubtitleTranslator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", chunk_size: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chunk_size = chunk_size
        self.system_prompt = """你是一个专业的字幕翻译员。请将给定的英文字幕翻译成中文。
要求：
1. 保持专业、准确、自然的翻译风格
2. 直接返回翻译结果，不要有任何解释
3. 返回格式必须是 JSON 数组，每个元素包含原文索引(index)和翻译(translation)
4. 翻译要简明扼要，符合中文表达习惯

例如输入：
1. Hello world
2. How are you?

应该输出：
[
    {"index": 1, "translation": "你好世界"},
    {"index": 2, "translation": "你好吗？"}
]"""

    def _create_translation_prompt(self, entries: List[SubtitleEntry]) -> str:
        prompt = ""
        for entry in entries:
            prompt += f"{entry.index}. {entry.content.strip()}\n"
        return prompt

    def translate_chunk(self, entries: List[SubtitleEntry]) -> List[Tuple[int, str]]:
        prompt = self._create_translation_prompt(entries)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        try:
            translations = json.loads(response.choices[0].message.content)
            return [(t["index"], t["translation"]) for t in translations]
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Translation format error: {e}")

    def translate_file(self, input_file: str, output_file: str):
        # 解析输入文件
        entries = SrtParser.parse(input_file)

        # 按chunk_size分组处理
        translated_entries = []
        for i in range(0, len(entries), self.chunk_size):
            chunk = entries[i:i + self.chunk_size]
            translations = self.translate_chunk(chunk)

            # 将翻译结果与原时间戳合并
            for original_entry, (index, translation) in zip(chunk, translations):
                translated_entry = SubtitleEntry(
                    index=original_entry.index,
                    timestamp=original_entry.timestamp,
                    content=translation
                )
                translated_entries.append(translated_entry)

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in translated_entries:
                f.write(str(entry) + '\n')


def main():
    # 使用示例
    translator = SubtitleTranslator(
        api_key="your-api-key",
        model="gpt-3.5-turbo",
        chunk_size=5  # 每次翻译5个字幕
    )

    translator.translate_file(
        input_file="input.srt",
        output_file="output_zh.srt"
    )


if __name__ == "__main__":
    main()
