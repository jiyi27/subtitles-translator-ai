import re
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple
from openai import OpenAI
import json
from pathlib import Path


class SubtitleError(Exception):
    """字幕处理相关的异常"""
    pass


class TranslationError(Exception):
    """翻译相关的异常"""
    pass


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
        if not Path(file_path).exists():
            raise SubtitleError(f"找不到字幕文件: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            raise SubtitleError(f"无法读取字幕文件，请确保文件编码是UTF-8: {file_path}")

        subtitle_blocks = re.split(r'\n\n+', content.strip())
        if not subtitle_blocks:
            raise SubtitleError("字幕文件为空")

        entries = []
        for block in subtitle_blocks:
            lines = block.split('\n')
            if len(lines) < 3:
                raise SubtitleError(f"格式错误的字幕块: {block}")

            index = int(re.sub(r'\D', '', lines[0].strip()))
            timestamp = lines[1]
            content = '\n'.join(lines[2:])
            entries.append(SubtitleEntry(index, timestamp, content))

        return entries


class SubtitleTranslator:
    def __init__(self, api_key: str, model: str = "gpt-4-0125-preview", chunk_size: int = 10):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chunk_size = chunk_size
        self.system_prompt = """你是一个专业的字幕翻译员, 请将给定的英文字幕翻译成中文。
要求：
1. 保持专业、准确、自然的翻译风格
2. 直接返回翻译结果，不要有任何解释
3. 返回格式必须是 JSON 数组，每个元素包含原文索引(index)和翻译(translation)
4. 翻译要简明扼要，符合中文表达习惯, 意译而非直译

例如输入：
3. Hello world
4. How are you?

应该输出：
[
    {"index": 3, "translation": "你好世界"},
    {"index": 4, "translation": "你好吗？"}
]"""

    @staticmethod
    def _format_subtitle_entries(entries: List[SubtitleEntry]) -> str:
        formatted_text = ""
        for entry in entries:
            formatted_text += f"{entry.index}. {entry.content.strip()}\n"
        return formatted_text

    def translate_subtitle_entry_chunk(self, entries: List[SubtitleEntry]) -> List[Tuple[int, str]]:
        if not entries:
            return []

        prompt = self._format_subtitle_entries(entries)
        try:
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
            except json.JSONDecodeError as e:
                raise TranslationError(f"JSON解析失败: {str(e)}, 位置: {entries[0].index}")
        except Exception as e:
            raise TranslationError(f"翻译失败: {str(e)}")

    def translate_file(self, input_file: str, output_file: str):
        entries = SrtParser.parse(input_file)

        total_chunks = (len(entries) + self.chunk_size - 1) // self.chunk_size
        translated_entries = []

        for chunk_index in range(0, len(entries), self.chunk_size):
            chunk = entries[chunk_index:chunk_index + self.chunk_size]
            current_chunk = chunk_index // self.chunk_size + 1
            print(f"翻译进度: {current_chunk}/{total_chunks}")

            translations = self.translate_subtitle_entry_chunk(chunk)
            for original_entry, (index, translation) in zip(chunk, translations):
                translated_entry = SubtitleEntry(
                    index=original_entry.index,
                    timestamp=original_entry.timestamp,
                    content=translation
                )
                translated_entries.append(translated_entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in translated_entries:
                f.write(str(entry) + '\n')
        print(f"\n翻译完成! 已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='字幕翻译工具')
    parser.add_argument('-i', '--input', required=True, help='输入的字幕文件路径 (例如: input.srt)')
    parser.add_argument('-o', '--output', required=True, help='输出的翻译后字幕文件路径 (例如: output_zh.srt)')
    args = parser.parse_args()

    try:
        translator = SubtitleTranslator(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-0125-preview",
            chunk_size=10
        )
        translator.translate_file(args.input, args.output)
    except SubtitleError as e:
        print(f"字幕处理错误: {str(e)}")
        sys.exit(1)
    except TranslationError as e:
        print(f"翻译错误: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()