import  h5py
import struct
import  numpy as np
import pathlib
from mio import MioWriter
import torch
import  time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_h5', required=True)
    parser.add_argument('--output_path', required=True, help='path to save mio feature')
    args = parser.parse_args()

    # save path
    root = pathlib.Path(args.output_path)

    # input h5 features
    f = h5py.File(args.input_h5, 'r')

    print('finish load h5py !!!')

    features = f['features'][:]

    # if required, please add to the mio
    # boxes = f['boxes'][:]  # 4, 36

    h5_ids = f['ids'][:].tolist()

    with MioWriter("VQACP2/trainval_features") as m:
        # for i, (box, fea) in enumerate(zip(boxes, features)):
        for i, fea in enumerate(features):
            print(f"{i}/{len(fea)}", fea.shape)
            with m.create_collection() as c:
                c.set_meta(struct.pack("<I", h5_ids[i]))
                # c.add_object(box.tobytes()) # if require bounding box, unlock it
                c.add_object(fea.tobytes())


if __name__ == '__main__':
    main()



#################### test ###################

# from mio import MIO

# m = MIO("./trainval_features_with_boxes")

# ids = []
# for i in range(m.size):
#     id_= struct.unpack("<I", m.get_collection_metadata(i))[0]
#     ids.append(id_)

# start= time.perf_counter()

# for i,id_ in enumerate(ids):
#     # print(i, len(ids))
#     true_id = ids.index(id_)
#     box = m.fetchone(colletion_id=true_id, object_id=0)
#     box=np.frombuffer(box, dtype=np.float32).reshape(4, 36)
#     feature = m.fetchone(colletion_id=true_id, object_id=1)
#     tensor = np.frombuffer(feature, dtype=np.float32).reshape(2048, 36)
#     features_h5 = features[true_id]
#     boxes_h5 = boxes[true_id]
#     if not (boxes_h5 == box).all():
#         print(id_)
#         print(i)

    # print(box.shape)
    # print(tensor.shape)
    # break
    # if i > 1000:
    #     break
    # h5_id = h5_ids.index(id_)
    # tensor_h5 = torch.from_numpy(features[h5_id]).permute(1,0)
    # assert (tensor==tensor_h5).all(), 'Wrong!!!'
    # assert torch.isclose(tensor, tensor_h5), 'Wrong!!!'

# print(start-very_start)
# print(time.perf_counter()-start)

