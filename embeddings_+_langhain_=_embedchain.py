# -*- coding: utf-8 -*-
"""Embeddings + Langhain = Embedchain

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cf5UCOx0xK6XKZcyI1DNlkqTJmrSY3lh

More details - https://github.com/embedchain/embedchain

Install embedchain library
"""

! pip install -q embedchain

"""# Setup OpenAI Key"""

import os
os.environ['OPENAI_API_KEY'] = 'sk-xxx' #environment variables

"""# Load embedchain"""

from embedchain import App

"""# Instantiate your BOT App"""

naval_bot = App()

"""# Add Online Resources"""

# Embed Online Resources
naval_bot.add("youtube_video", "https://www.youtube.com/watch?v=3qHkcs3kG44")
naval_bot.add("pdf_file", "https://navalmanack.s3.amazonaws.com/Eric-Jorgenson_The-Almanack-of-Naval-Ravikant_Final.pdf")
naval_bot.add("web_page", "https://nav.al/feedback")
naval_bot.add("web_page", "https://nav.al/agi")

"""Add Local Questions"""

# Embed Local Resources
naval_bot.add_local("qna_pair", ("Who is Naval Ravikant?", "Naval Ravikant is an Indian-American entrepreneur and investor."))

"""# Start Questioning!"""

print(naval_bot.query("What are the ways to become rich?"))

print(naval_bot.query("What do you think about fame and wealth?"))

print(naval_bot.query("What is a better alternative to UBI?"))

