#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def _tag_to_open_close(tag: str) -> tuple:
    """return html opening and closing tag strings <tag> and </tag>

    Args:
        tag (str): tag of an html element

    Returns:
        tuple: <tag>, </tag>
    """
    return f'<{tag}>', f'</{tag}>'


def get_different_tags(text: str) -> set:
    """Search and return all the different html tags in a string

    Args:
        text (str): text to search the html tags

    Returns:
        set: found html tags
    """
    tags_in = set()
    tags_out = set()
    index = text.find('<')
    i = 0
    while index != -1:
        i += 1
        if i > 100:
            break
        index_end = text.find('>', index)
        if index_end == -1:
            break
        tag = text[index+1:index_end]  # tag value
        index = text.find('<', index_end)
        if len(tag) == 0:
            continue
        if tag[0] == '/':  # tag closing an element
            tag = tag[1:]
            # text is search linearily
            # -> check if closing tags are in opening tags list
            if tag in tags_in:
                tags_out.add(tag)
        else:  # tag opening an element
            tags_in.add(tag)
    # filter tags to keep those paired
    # tags = set(tag for tag in tags_in if tag in tags_out)
    # return tags
    return tags_out


def remove_element(text: str, tag: str) -> str:
    """remove html element from a text

    Args:
        text (str): input text
        tag (str): html element tag

    Returns:
        str: output text
    """
    tag, tag_end = _tag_to_open_close(tag)
    index = text.find(tag)
    while index != -1:
        index_end = text.find(tag_end, index) + len(tag_end)
        text = text[:index] + text[index_end:]
        index = text.find(tag)
    return text


def extract_elements(text: str, tag: str) -> list:
    """extract html elements corresponding to a html tag in a text

    Args:
        text (str): input text
        tag (str): html tag

    Returns:
        list: list of elements found
    """
    tag, tag_end = _tag_to_open_close(tag)
    out = []
    index = text.find(tag)
    while index != -1:
        index_end = text.find(tag_end, index)
        out.append(text[index+len(tag):index_end])
        index = text.find(tag, index_end)
    return out


# %% END OF FILE
###
