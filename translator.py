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
    def __init__(self, api_key: str, model: str = "gpt-4o", chunk_size: int = 10):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chunk_size = chunk_size
        self.system_prompt = """
        你是一个专业的中文翻译官, 按照以下要求翻译字幕:
        1. 根据英文内容直译, 保持原有格式
        2. 根据第一次直译的结果, 考虑上下文的语境, 遵守原意的前提下进行重新意译
        
        例如输入：
        3. We have thousands of friends on this.
        4. incredible movement.
        
        严格按照一下格式输出:
        {
          "literal": [
            {
              "index": 3,
              "translation": "世界, 你好"
            },
            {
              "index": 4,
              "translation": "你好吗?"
            }
          ],
          "free": [
            {
              "index": 3,
              "translation": "你好吗? 世界"
            },
            {
              "index": 4,
              "translation": "最近过的怎么样?"
            }
          ]
        }
        """

    @staticmethod
    def _format_subtitle_entries(entries: List[SubtitleEntry]) -> str:
        formatted_text = ""
        for entry in entries:
            formatted_text += f"{entry.index}. {entry.content.strip()}\n"
        return formatted_text

    @staticmethod
    def _clean_response(response: str) -> str:
        # 移除开头和结尾的空白字符
        cleaned = response.strip()

        # 移除markdown标记
        if cleaned.startswith('```'):
            # 寻找第一个换行符，移除整个```json或```python等行
            first_newline = cleaned.find('\n')
            if first_newline != -1:
                cleaned = cleaned[first_newline:].strip()
            # 移除结尾的```
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()

        return cleaned

    def translate_subtitle_entry_chunk(self, entries: List[SubtitleEntry]) -> List[Tuple[int, str]]:
        prompt = self._format_subtitle_entries(entries)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_response = response.choices[0].message.content
            cleaned_response = self._clean_response(raw_response)

            try:
                translations = json.loads(cleaned_response)
                return [(t["index"], t["translation"]) for t in translations["free"]]
            except (json.JSONDecodeError, KeyError) as e:
                # return [(entry.index, entry.content) for entry in entries]
                raise TranslationError(f"解析响应失败, 请检查响应格式是否正确:\n{cleaned_response}\n" + str(e))
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
            model="gpt-4o",
            chunk_size=20
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