import argparse
import io
import itertools

import pandas as pd
from sklearn import linear_model

def readXYZFiles(files):
    xyzdfs = []
    for xyzfile in files:
        with open(xyzfile, mode='r') as f:
            # 3行目以降が必要なのでそこから取り出す
            xyzdata = f.readlines()[2:]
        # stripで末端の改行コードを取り除く
        xyzdata = [s.strip() for s in xyzdata]
        # 空行は取り除く
        xyzdata = [s for s in xyzdata if s != '']
        # pandasのDataFrameに変換する
        xyzdf = pd.read_csv(io.StringIO(xyzdata), header=None, delim_whitespace=True,
                            names=['elementSymbol','x','y','z'],
                            dtype={'elementSymbol':str})
        xyzdfs.append(xyzdf)

    return xyzdfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ovwmols : Overwrite Molecule Files.')

    parser.add_argument('--format', help='molecule file format', required=True, choices=['XYZ'], type=str)
    parser.add_argument('--ref', help='conversion reference atoms(0-based)', default=[], nargs='*')
    parser.add_argument('--file', help='molecule file', required=True, nargs='*')

    # 引数解析
    args = parser.parse_args()
    refatoms = [int(s) for s in args.ref]

    # 各ファイルを読み込みDataFrameに変換する
    xyzdfs = readXYZFiles(args.file)

    # 各ファイル同士で座標変換による重ね合わせを行い、
    # 最も他のファイルとの一致が良いファイルを後の基準ファイルにする
    for xyzdf_i, xyzdf_j in itertools.combinations(xyzdfs, 2):
        if len(refatoms) != 0:
            xyzdf_i_refs = xyzdf_i.iloc[refatoms]
            xyzdf_j_refs = xyzdf_j.iloc[refatoms]
        else:
            xyzdf_i_refs = xyzdf_i
            xyzdf_j_refs = xyzdf_j

        # TODO
        # assertとしてxyzdf_i_refsとxyzdf_j_refsの元素記号が一致しているか確認する

        # 姿勢推定
        # Rとtを求める

        # score評価、記録


