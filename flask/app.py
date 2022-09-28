from flask import Flask, request
from torch.utils.tensorboard import SummaryWriter

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def received_data():
    json = request.get_json()
    for writer_name, data in json.items():
        writer = SummaryWriter("../" + writer_name)
        for key, hist in data.items():
            count = 0
            for idx, event in enumerate(hist):
                for phase, value in event.items():
                    writer.add_scalars(key, {phase: value}, count)
                if (idx%2 == 1):
                    count += 1
    return 'OK'

if __name__ == '__main__':
    app.run(debug = True)
