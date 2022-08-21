import pyaudio
from pydub import AudioSegment


def play_audio_background(audio_file):

    audio_file = str(audio_file).replace('.mp3', '.wav')
    # CHUNK_SIZE = 4096
    # file_size = os.path.getsize(audio_file)

    sound = AudioSegment.from_file(audio_file)
    player = pyaudio.PyAudio()

    stream = player.open(format=player.get_format_from_width(sound.sample_width),
                         channels=sound.channels,
                         rate=sound.frame_rate,
                         output=True)

    with open(audio_file, 'rb') as fh:
        #while fh.tell() != file_size:
        #AUDIO_FRAME =  ##直接读全部，减少噪音
        data = fh.read()
        stream.write(data)


def test_play():
    play_audio_background('/Users/qimick/doc/@todo/opencv/citibai/speaker/audio/gtts/begin.mp3')


if __name__ == '__main__':
    from multiprocessing import Process
    p = Process(target=test_play, args=())
    p.start()
    p.join()
    #test_play()