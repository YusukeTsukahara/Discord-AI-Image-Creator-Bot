import discord
from dotenv import load_dotenv
import os
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import cv2
import numpy as np
import asyncio

# .envファイルから環境変数を読み込む
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
TEXT_MODEL_ID = os.getenv('TEXT_MODEL_ID')
IMAGE_MODEL_ID = os.getenv('IMAGE_MODEL_ID')
DEVICE = os.getenv('DEVICE')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# パイプラインの作成
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(IMAGE_MODEL_ID, use_auth_token=HUGGINGFACE_TOKEN)
img2img_pipe = img2img_pipe.to(DEVICE)

txt2img_pipe = StableDiffusionPipeline.from_pretrained(TEXT_MODEL_ID, use_auth_token=HUGGINGFACE_TOKEN)
txt2img_pipe = txt2img_pipe.to(DEVICE)

# 線画に変換する関数を作成
def convert_to_sketch(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv_gray_image = cv2.bitwise_not(gray_image)
    blur_image = cv2.GaussianBlur(inv_gray_image, (21, 21), sigmaX=0, sigmaY=0)
    inv_blur_image = cv2.bitwise_not(blur_image)
    sketch_image = cv2.divide(gray_image, inv_blur_image, scale=256.0)
    return sketch_image

# Discord Botを起動するための設定をする
intents = discord.Intents.default()
intents.messages = True
intents.typing = False
intents.presences = False
intents.message_content = True
client = discord.Client(intents=intents)

# Botが起動した時にメッセージを表示する
@client.event
async def on_ready():
    print('起動完了')

# PIL画像をリサイズして、そのサイズが8で割り切れるようにする関数を追加
def resize_image_to_multiple_of_eight(image):
    width, height = image.size
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    return image.resize((new_width, new_height), Image.LANCZOS)

# 処理を実行する関数を定義
async def process_message(message, prompt, negative_prompt, convert_to_line_drawing, guidance_scale, num_images_per_prompt):
    try:
        if message.attachments:
            # 添付ファイルがある場合、img2imgパイプラインを使用
            attachment = message.attachments[0]
            await attachment.save('input_image.png')

            # 画像を読み込む
            init_image = Image.open('input_image.png').convert('RGB')
            
            if init_image:
                # 画像を8で割り切れるようにリサイズ
                init_image = resize_image_to_multiple_of_eight(init_image)

            # PIL画像をnumpy配列に変換
            init_image_np = np.array(init_image)

            # pipe関数を呼び出し
            result = img2img_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, image=init_image_np)
        else:
            # 添付ファイルがない場合、txt2imgパイプラインを使用
            result = txt2img_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=30, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt)

        # 生成された画像を保存する
        generated_image_paths = []

        for idx, image in enumerate(result.images):
            generated_image_path = f'output_{idx}.png'
            image.save(generated_image_path)  # 画像をファイルに保存する
            generated_image_paths.append(generated_image_path)

        # 生成された元の画像をDiscordに送信する
        for generated_image_path in generated_image_paths:
            with open(generated_image_path, 'rb') as f:
                original_picture = discord.File(f, filename='generated_image.png')
                await message.channel.send(file=original_picture)

            if convert_to_line_drawing:
                # 生成された画像を線画に変換して送信する
                sketch_image = convert_to_sketch(generated_image_path)
                sketch_image_path = f'output_sketch_{idx}.png'
                cv2.imwrite(sketch_image_path, sketch_image)

                with open(sketch_image_path, 'rb') as f:
                    sketch_picture = discord.File(f, filename='sketch_image.png')
                    await message.channel.send(file=sketch_picture)

    except Exception as e:
        await message.channel.send(f'エラーが発生しました: {str(e)}')

# ユーザーからメッセージが送信された時の処理を設定する
@client.event
async def on_message(message):

    # Bot自身からのメッセージは無視する
    if message.author == client.user:
        return

    # '!img [入力文字列]'という形式のメッセージが送信された場合
    if message.content.startswith('!img'):
        prompt = message.content[5:].strip()
        
        # ネガティブプロンプトも同様に単一の文字列として定義
        negative_prompt = 'painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime'
        
        # デフォルトのプロンプト
        default_prompt ='(ultra Details high definition:1.2),(ultra Details high definition video:1.2),(ultra Details high resolution:1.2).(Next generation video standard:1.2),(ultra Details extremely very sophisticated photo reality:1.2),(Break),,,,, ((yout:1.3)(innocence:1.2) (young:1.2) (pretty:1.2)(cute:1.2)(kawaii:1.4):1.5),(gentle girl:1.2),(break),,,,, (dynamic angle:1.1),(multi angle:1.1),(break),,,,, (nostalgic color light:1.4),(retro Nostalgic taste:1.2),(anime coating illustration:1.2),(Glossy line drawing solid cel painting:1.2),(colorful luster:1.2),(break),,,,,, (Light and shadow),(Dirt and scratch details),(break),,,,,, (Movie style drawing),(Original picture),(fantastic finale:1.2),(mysterious summarize:1.2),(break),,,,,,'

        # プロンプト内に-LineDrawingが含まれているか確認
        convert_to_line_drawing = '-LineDrawing' in prompt
        prompt = prompt.replace('-LineDrawing', '').strip()

        # guidance_scaleとnum_images_per_promptの初期値を設定
        guidance_scale = 7.5
        num_images_per_prompt = 1

        # プロンプトから-GuidanceScaleと-NumImagesPerPromptの値を抽出
        if '-GuidanceScale' in prompt:
            try:
                guidance_scale = float(prompt.split('-GuidanceScale')[1].split()[0])
                prompt = prompt.replace(f'-GuidanceScale {guidance_scale}', '').strip()
            except (IndexError, ValueError):
                pass

        if '-NumImagesPerPrompt' in prompt:
            try:
                num_images_per_prompt = int(prompt.split('-NumImagesPerPrompt')[1].split()[0])
                prompt = prompt.replace(f'-NumImagesPerPrompt {num_images_per_prompt}', '').strip()
            except (IndexError, ValueError):
                pass

        await message.channel.send('画像生成を開始しました(生成完了まで時間かかります)')

        try:
            # 15分のタイムアウトを設定してメッセージ処理を実行
            await asyncio.wait_for(process_message(message, prompt + default_prompt, negative_prompt, convert_to_line_drawing, int(guidance_scale), int(num_images_per_prompt)), timeout=3600)
        except asyncio.TimeoutError:
            await message.channel.send('処理がタイムアウトしました。再度お試しください。')
        except Exception as e:
            await message.channel.send(f'エラーが発生しました: {str(e)}')

# Discord Botを起動する
client.run(DISCORD_TOKEN)