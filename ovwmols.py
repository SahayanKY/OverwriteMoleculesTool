import argparse
import io
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, cdist, pdist

def readFiles(files, fileformat):
    """
    指定されたフォーマットに合わせてファイルを読み込み、
    DataFrameのリストにする
    DataFrameの列名はelementSymbol, x, y, zの4つである
    """
    if fileformat == 'XYZ':
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
            xyzdf = pd.read_csv(io.StringIO('\n'.join(xyzdata)), header=None, delim_whitespace=True,
                                names=['elementSymbol','x','y','z'],
                                dtype={'elementSymbol':str})
            xyzdfs.append(xyzdf)
    else:
        raise ValueError('Unsupported file format')

    return xyzdfs

def estimate_conversionParameter(X, Y):
    """
    https://www.slideshare.net/ttamaki/20090924
    X: previous coordinates (n*3 matrix)
    Y: current coordinates  (n*3 matrix)

    Y ~ X @ R.T + t
    となるようなRとtを求める
    """
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    X_shift = X - X_mean
    Y_shift = Y - Y_mean

    # (3*n).(n*3)の内積をとる
    # 特異値分解を行う行列を計算
    W = X_shift.T @ Y_shift

    # Wを特異値分解
    U, Sigma, VT = np.linalg.svd(W)

    # 回転行列Rを計算
    detVU = np.linalg.det(VT @ U)
    R = VT.T @ np.diag([1,1, detVU]) @ U.T

    # 並進移動tを計算
    # Y_mean = X_mean @ R.T + tより計算
    t = Y_mean - X_mean @ R.T

    return R, t

def getConversionParameterListRandom(xyzdfs, refAtomIndexes):
    numFile = len(xyzdfs)

    xyzdf_0 = xyzdfs[0]
    xyzdf_0_refs =xyzdf_0.iloc[refAtomIndexes]
    # ref内で1つしかない元素を探す
    num_ref_symbol = xyzdf_0_refs['elementSymbol'].value_counts()
    onlyone_ref_symbolList = num_ref_symbol[num_ref_symbol==1].index.tolist()


    #xyzdf_1でonlyonesymbol原子とそれ以外の原子間の距離行列を計算
    #xyzdf_0_refsの距離行列と比較し、xyzdf_1内のxyzdf_0_refsと対応する原子を絞り込む
    #
    #
    #xyzdf_0_refsとxyzdf_1内の原子の数を元素毎に求める
    #indexが元素記号、値が各元素の原子数のSeriesになる
    xyzdf_0_refs_numEachElements = xyzdf_0_refs['elementSymbol'].value_counts().sort_index()
    xyzdf_1_numEachElements = xyzdf_1['elementSymbol'].value_counts().sort_index()


    if len(onlyone_ref_symbolList) > 0:
        #TODO ここ、[0]じゃなくて、xyzdf_0にもxyzdf_1にも少ない元素がいいね
        ele = onlyone_ref_symbolList[0]

        #indexがeleである行を除く
        #indexが'C'である行を除いたseriesが返る
        #testseries.loc[~(testseries.index=='C')]
        xyzdf_0_refs_numEachElements_except = xyzdf_0_refs_numEachElements.loc[~(xyzdf_0_refs_numEachElements.index==ele)]
        xyzdf_1_numEachElements_except = xyzdf_1_numEachElements.loc[~(xyzdf_1_numEachElements.index==ele)]

        #refdis = squareform(pdist(xyzdf0_ref[['x','y','z']]))[2]
        #refdis = np.delete(refdis,2)
        #xyzdf_1_Odis = cdist(xyzdf_1[xyzdf_1['elementSymbol']=='O'][['x','y','z']], xyzdf_1[xyzdf_1['elementSymbol']!='O'][['x','y','z']])
        #

    for j in range(1,numFile):
        xyzdf_j = xyzdfs[j]




    pass

def getConversionParameterList(xyzdfs, refAtomIndexes):
    """
    各ファイルを一致させるための変換行列を得る
    """
    numFile = len(xyzdfs)

    # 各ファイル同士で座標変換による重ね合わせを行い、
    # 最も他のファイルとの一致が良いファイルを後の基準ファイルにする
    scorelist = [[0]*numFile for i in range(numFile)]
    rotateMatrixList = [[0]*numFile for i in range(numFile)]
    transMatrixList = [[0]*numFile for i in range(numFile)]
    for i, j in itertools.combinations(range(numFile), 2):
        # i < j
        xyzdf_i = xyzdfs[i]
        xyzdf_j = xyzdfs[j]

        if len(refAtomIndexes) != 0:
            xyzdf_i_refs = xyzdf_i.iloc[refAtomIndexes]
            xyzdf_j_refs = xyzdf_j.iloc[refAtomIndexes]
        else:
            xyzdf_i_refs = xyzdf_i
            xyzdf_j_refs = xyzdf_j

        # TODO
        # assertとしてxyzdf_i_refsとxyzdf_j_refsの元素記号が一致しているか確認する

        # DataFrameとしてはここでは使わないのでndarrayに変換
        xyz_i_refs = xyzdf_i_refs[['x','y','z']].values
        xyz_j_refs = xyzdf_j_refs[['x','y','z']].values
        # 変換パラメータ推定
        # Rとtを求める
        R, t = estimate_conversionParameter(xyz_i_refs, xyz_j_refs)
        rotateMatrixList[i][j] = R
        transMatrixList[i][j] = t

        # Rとtを使ってxyzdf_iからxyzdf_jを予測し、一致具合を二乗和で評価
        # n*3行列
        errorxyz = xyz_j_refs - (xyz_i_refs @ R.T + t)
        # score評価、記録
        score = np.sum(np.power(errorxyz,2))
        scorelist[i][j] = score
        scorelist[j][i] = score

    # scorelistの各行ごとにscoreの和をとる
    # -> 最も他との一致がいいファイル(基準ファイルreffileindex)(最も総和値が小さい行)を探す
    scorelist = np.sum(scorelist, axis=1)
    reffileindex = np.argmin(scorelist)

    # rotateMatrixListとtransMatrixListと基準ファイル(reffile)を使って
    # 他のファイルの座標を
    # 基準ファイルの座標に重なるように変換するRとtを求め、返す
    conversionParameterList = []
    for j, xyzdf_j in enumerate(xyzdfs):
        if j == reffileindex:
            # この場合は恒等変換
            R = np.diag([1,1,1])
            t = np.array([0,0,0])
        elif j > reffileindex:
            # [reffileindex][j]から変換行列を取り出す
            # この変換行列はreffileindex -> j への変換なので逆変換にする
            R = rotateMatrixList[reffileindex][j]
            t = transMatrixList[reffileindex][j]
            t = -t @ R
            R = R.T
        else:
            # j < reffileindex
            # この変換行列はj -> reffileindexへの変換
            R = rotateMatrixList[j][reffileindex]
            t = transMatrixList[j][reffileindex]
        conversionParameterList.append([R,t])
    return conversionParameterList

def main(args):
    # 各ファイルを読み込みDataFrameのリストに変換する
    xyzdfs = readFiles(args.file, args.format)
    # 一致させる基準原子のインデックスを取得
    refAtomIndexes = [int(s) for s in args.ref]

    # 各ファイルを一致させるための変換行列を取得する
    conversionParameterList = getConversionParameterList(xyzdfs, refAtomIndexes)

    # 各ファイルを一致させる(変換)
    # 変換後のデータはリストに記録していき、最後にファイル出力する
    convertedxyzdata = []
    for xyzdf_i, (R, t) in zip(xyzdfs,conversionParameterList):
        xyz_i_converted = xyzdf_i[['x','y','z']].values @ R.T + t
        convertedxyzdata.extend(['{} {} {} {}'.format(s,x,y,z) for s,(x,y,z) in zip(xyzdf_i['elementSymbol'], xyz_i_converted)])

    # 変換結果をxyzファイルに書き出し
    with open(args.save, mode='w') as f:
        f.write('\n'.join(convertedxyzdata))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ovwmols : Overwrite Molecule Files.')

    parser.add_argument('--format', help='molecule file format', required=True, choices=['XYZ'], type=str)
    parser.add_argument('--ref', help='conversion reference atoms(0-based)', default=[], nargs='*')
    parser.add_argument('--file', help='molecule file', required=True, nargs='*')
    parser.add_argument('--save', help='save destination', default='ovwmols.xyz', type=str)

    # 引数解析
    # main実行
    main(parser.parse_args())


