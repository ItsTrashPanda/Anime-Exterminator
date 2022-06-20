import discord
import tensorflow
import numpy as np
from tensorflow import keras
import cv2

guild_1 = discord.Guild

intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_member_join(member):
    img = member.avatar_url_as(format='png',static_format='png',size=64)
    await img.save('./avatars/avatar.png')
    status = process_image('./avatars/avatar.png')
    if(status == 0):
        await member.kick(reason='We don\'t like weebs around here. Please come back when you\'ve changed your avatar. Thank you, have a nice day! :)')
        #await channel.send(member.display_name + ' was a weeb! Thankfully they are gone now, phew that was a close one!')
    elif(status == 1):
        print('you good')
        #await channel.send(member.display_name + ', nice avatar!')

def process_image(image):
    im = []
    new_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Grabs images and converts them to greyscale
    new_image = cv2.resize(new_image, (64, 64)) 
    new_image = cv2.Sobel(new_image, cv2.CV_64F, 1, 0, ksize=5)
    im.append(new_image)
    im = np.array(im).reshape(-1, 64, 64)
    im = im / 255.0
    model = keras.models.load_model('./model')
    for i in model.predict(im):
        print(np.argmax(i))
        return(np.argmax(i))



client.run('')