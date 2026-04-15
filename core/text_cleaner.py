import charset_normalizer
import unicodedata
import re


def clean_text(raw_bytes, encoding=None, enable_cleaning=False):
    """
    核心文本处理逻辑
    接收二进制数据，返回 (processed_str, raw_str, encoding_name, stats)。

    encoding       : 指定编码名称（如 'gb18030'、'utf-8'）时直接解码，
                     为 None 时使用 charset_normalizer 自动检测。
    enable_cleaning: True 时执行脏字符清除 + Unicode NFKC 标准化。
    """
    # 1. 解码
    if encoding:
        try:
            raw_str = raw_bytes.decode(encoding, errors='replace')
            detected_encoding = encoding
        except LookupError:
            return None, f"未知编码名称「{encoding}」，请检查拼写", None, {}
        except Exception as e:
            return None, f"使用编码「{encoding}」解码失败：{e}", None, {}
    else:
        result = charset_normalizer.from_bytes(raw_bytes).best()
        if not result:
            return None, "charset_normalizer 无法识别该文件的编码", None, {}
        detected_encoding = result.encoding
        raw_str = str(result)

    # 2. 统计原始信息
    stats = {
        "detected_encoding": detected_encoding,
        "original_length": len(raw_str),
        "cleaning_enabled": enable_cleaning,
    }

    if enable_cleaning:
        dirty_pattern = re.compile(
            r'[\u200b-\u200f\u202a-\u202e\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
        )
        cleaned_str, dirty_count = dirty_pattern.subn('', raw_str)
        stats["dirty_chars_removed"] = dirty_count
        processed_str = unicodedata.normalize('NFKC', cleaned_str)
    else:
        processed_str = raw_str
        stats["dirty_chars_removed"] = 0

    stats["final_length"] = len(processed_str)

    return processed_str, raw_str, detected_encoding, stats
