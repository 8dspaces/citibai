from gtts import gTTS
import pathlib
from pydub import AudioSegment

from brain.words import sounds
from speaker.player import play_audio_background as play1
from speaker.playerLite import play_audio_background as play2


convert_wav = True


def generate_audio(pth="gtts"):

    audio_dir = pathlib.Path(__file__).parent.absolute().joinpath('audio/{}'.format(pth))

    for name, sound in sounds.items():

        audio_file = audio_dir.joinpath(name + '.mp3')
        tts = gTTS(text=sound, lang='zh-CN')
        tts.save(audio_file)

        if convert_wav:
            translate_to_wav(audio_file)


def translate_to_wav(src):

    target = str(src).replace('mp3', 'wav')
    sound = AudioSegment.from_mp3(src)

    sound.export(target, format="wav")


def test_play_audio(name, pth="gtts"):

    audio_dir = pathlib.Path(__file__).parent.absolute().joinpath('audio/{}'.format(pth))
    #playsound(audio_dir.joinpath(name + '.mp3'))
    play1(audio_dir.joinpath(name + '.mp3'))
    play2(audio_dir.joinpath(name + '.mp3'))


if __name__ == '__main__':

    generate_audio()
    test_play_audio('cooperate')