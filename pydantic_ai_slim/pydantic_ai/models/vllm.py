from __future__ import annotations as _annotations

import base64
from typing import TYPE_CHECKING, Any

from ..messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    UserPromptPart,
    VideoUrl,
)
from . import download_item
from .openai import OpenAIChatModel

from openai.types import chat
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.chat.chat_completion_content_part_param import File, FileFile

if TYPE_CHECKING:
    pass


class VLLMChatModel(OpenAIChatModel):
    """A model that uses the vLLM API (OpenAI-compatible) with extended capabilities like VideoUrl."""

    async def _map_user_prompt(self, part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:  # noqa: C901
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url: ImageURL = {'url': item.url}
                    if metadata := item.vendor_metadata:
                        image_url['detail'] = metadata.get('detail', 'auto')
                    if item.force_download:
                        image_content = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url['url'] = image_content['data']
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, VideoUrl):
                    # Support for vLLM/Qwen video input
                    # Note: We bypass type checking here because 'video_url' is not yet in standard OpenAI types and is unlikely to be included
                    content.append({'type': 'video_url', 'video_url': {'url': item.url}})  # type: ignore
                elif isinstance(item, BinaryContent):
                    if self._is_text_like_media_type(item.media_type):
                        content.append(
                            self._inline_text_file_part(
                                item.data.decode('utf-8'),
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    elif item.is_image:
                        image_url = ImageURL(url=item.data_uri)
                        if metadata := item.vendor_metadata:
                            image_url['detail'] = metadata.get('detail', 'auto')
                        content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    elif item.is_audio:
                        assert item.format in ('wav', 'mp3')
                        audio = InputAudio(data=base64.b64encode(item.data).decode('utf-8'), format=item.format)
                        content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                    elif item.is_document:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=item.data_uri,
                                    filename=f'filename.{item.format}',
                                ),
                                type='file',
                            )
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    downloaded_item = await download_item(item, data_format='base64', type_format='extension')
                    assert downloaded_item['data_type'] in (
                        'wav',
                        'mp3',
                    ), f'Unsupported audio format: {downloaded_item["data_type"]}'
                    audio = InputAudio(
                        data=downloaded_item['data'], format=downloaded_item['data_type']  # type: ignore
                    )
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):
                    # For now, we only support downloading the document and sending it as a file
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    content.append(
                        File(
                            file=FileFile(
                                file_data=downloaded_item['data'],
                                filename=f'filename.{downloaded_item["data_type"]}',
                            ),
                            type='file',
                        )
                    )
                else:
                    # CachePoint is skipped here as it's not content
                    pass

        return {'role': 'user', 'content': content}
