from datasets import Dataset
from datasets import load_dataset
import pandas as pd
import ast
from PIL import Image, ImageDraw, ImageFont

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))

df = pd.DataFrame(sold_train)

filtered_df = df[df['label'] == 'OFF']

abusiveWords = []

for index, row in filtered_df.iterrows():
  words = row['tokens'].split(" ")
  rational = row['rationales']
  arr = ast.literal_eval(rational)
  for index, val in enumerate(arr):
    if(val==1):
      abusiveWords.append(words[index])
print(abusiveWords)

df = pd.DataFrame(abusiveWords, columns=['Abusive Words'])
df.to_csv("Abusive_words.csv")

font_path = "NotoSansSinhala-Black.ttf"  # Specify the path to the Sinhala font file
font_size = 48 
j = 0

for i in abusiveWords:
  for k in i:
    width, height = 500, 500  # Specify the dimensions of the image 
    background_color = (255, 255, 255)  # Specify the white background color
    new_image = Image.new('RGB', (width, height), background_color)

    text = k

    # Specify the font size for the text

    # Load the Sinhala font with the specified size
    font = ImageFont.truetype(font_path, font_size)

    draw = ImageDraw.Draw(new_image)

    # Calculate the position to center the text
    text_width, text_height = draw.textsize(text, font=font)
    text_position = ((width - text_width) // 2, (height - text_height) // 2)

    # Draw the Sinhala text on the image
    draw.text(text_position, text, font=font, fill=(0, 0, 0))  # Use black color for the text

    new_image.save("Images/output"+str(j)+".png")  # Save the modified image as PNG, or choose the desired image format
    j += 1
