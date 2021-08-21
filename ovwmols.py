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
    dfs = []
    if fileformat == 'XYZ':
        for xyzfile in files:
            with open(xyzfile, mode='r') as f:
                # 3行目以降が必要なのでそこから取り出す
                xyzdata = f.readlines()[2:]
            # stripで末端の改行コードを取り除く
            xyzdata = [s.strip() for s in xyzdata]
            # 空行は取り除く
            xyzdata = [s for s in xyzdata if s != '']
            # pandasのDataFrameに変換する
            df = pd.read_csv(io.StringIO('\n'.join(xyzdata)), header=None, delim_whitespace=True,
                                names=['elementSymbol','x','y','z'],
                                dtype={'elementSymbol':str})
            dfs.append(df)
    elif fileformat == 'GJF':
        for gjffile in files:
            with open(gjffile, mode='r') as f:
                gjfdata = f.readlines()
            # 分子指定セクションを取得
            # 2回目の空行の次からが分子指定セクション
            start, end = [i for i, s in enumerate(gjfdata) if re.match(r'^[ \t\n]+$', s)][1:3]
            # 空行の次(+1)は電荷・多重度の行なので、更に+1
            start += 2
            # 末端の改行コードを取り除く
            xyzdata = [s.strip() for s in gjfdata[start:end]]
            # pandasのDataFrameに変換する
            df = pd.read_csv(io.StringIO('\n'.join(xyzdata)), header=None, delim_whitespace=True,
                                names=['elementSymbol','x','y','z'],
                                dtype={'elementSymbol':str})
            dfs.append(df)
    else:
        raise ValueError('Unsupported file format')

    return dfs

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

    # Rとtを使ってXからYを予測し、一致具合を二乗和で評価
    # n*3行列
    errorxyz = Y - (X @ R.T + t)
    # score評価、記録
    score = np.sum(np.power(errorxyz,2))

    return R, t, score

def getCandidates_for_df1refsIndexes(df0, df1, df0refsIndexes, bruteForce):
    """
    df0とdf1を見比べて、df0refsに対応するdf1の原子はどれかの候補を挙げる
    Iteratorが返る
    df0, df1 : 元素記号でソートされているdf
    bruteForce : 総当たり式にするか
    """
    #TODO 引数df0からdf0refsに変えて、df0refsIndexesを除いてもいいんじゃないか？

    #後の原子の対応関係を調べる際に楽をするために元素記号でソートしておく
    df0 = dfs[0].sort_values('elementSymbol')
    df1 = dfs[1].sort_values('elementSymbol')
    #元素記号のソートの結果をrefAtomIndexesに反映する
    #内容は変わらないが、df0のデータの並び順に合うようになる
    refAtomIndexes = [i for i in df0.index if i in refAtomIndexes]
    #df0のインデックスの順番は変わっているのでlocで取得
    #refAtomIndexesを並び替えたので、df0refsも元素記号順になっている
    df0refs =df0.loc[refAtomIndexes]

    #
    #df0refsとdf1内の原子の数を元素毎に求める
    #後でdf0refsとdf1の原子間距離の比較をする際に、データを複製する必要があるのでその数を求める
    #また、df0refs内に1つしかない元素を探すのも目的
    #indexが元素記号、値が各元素の原子数のSeriesになる
    #value_counts()では多い順に元素が並んでしまうため、sort_index()で元素記号順に並べ直す
    df0refs_numEachElements = df0refs['elementSymbol'].value_counts().sort_index()
    df1_numEachElements = df1['elementSymbol'].value_counts().sort_index()

    # ref内で1つしかない元素を探す
    onlyone_ref_symbolList = df0refs_numEachElements[df0refs_numEachElements==1].index.tolist()


    #df0refsと対応するdf1内の原子を絞り込む
    if len(onlyone_ref_symbolList) > 0:
        #onlyone_ref_symbolListに含まれる元素の内、df1内でも少ない元素を選定 -> 元素記号(str)を取得
        ele = df1_numEachElements.loc[onlyone_ref_symbolList].idxmin()

        #df1中の各原子とele候補(df1内で1つだけとは限らない)の距離行列を計算
        #df0refsの距離行列と比較することでdf1の原子を絞る
        #
        #df0refs内でのeleとの距離行列を計算
        df0refs_eleIndex = np.where(df0refs['elementSymbol']==ele)
        df0refs_dis = squareform(pdist(df0refs[['x','y','z']]))[df0refs_eleIndex]
        #df1内でのele(候補)との距離行列を計算
        df1_dis = cdist(df1[df1['elementSymbol']==ele][['x','y','z']], df1[['x','y','z']])
        #

        #最終的にdf1candirefsを出す


    else:
        #絞りこむ手段がなかったため、総当たり式にする
        #TODO DataFrameで渡すかは上の実装が完了しないと分からない
        df1candirefs = df1


    return None


def getConversionParameter(df0, df1, df0refsIndexes, sorted, bruteForce=False):
    """
    df1を変換してdf0に重ねるようなRとtを求める
    sorted : df0とdf1の原子の順番が予め合わせられている場合
    df0refsIndexes : df0の基準原子のインデックス
    """

    if sorted:
        df1refsIndexesIterator = [df0refsIndexes]
    else:
        df1refsIndexesIterator = getCandidates_for_df1refsIndexes(df0, df1, df0refsIndexes, bruteForce)
    #TODO ソートの関係でilocが適切なのかlocが適切なのかよく考える

    df1xyz = df1[['x','y','z']]
    minimumScore = np.inf
    for df1refsIndexes in df1refsIndexesIterator:
        df1refsIndexes = list(df1refsIndexes)
        X = df1xyz.iloc[df1refsIndexes].values
        R, t, score = estimate_conversionParameter(X, Y)
        if score < minimumScore:
            minimumScore = score
            minimumdf1refsIndexes = df1refsIndexes
            minimumR = R
            minimumt = t

    return minimumR, minimumt, minimumdf1refsIndexes

def getMostSuitableConversionParameterList(dfs, refAtomIndexesList):
    """
    各ファイルを最も一致させるための変換行列を得る
    """
    numFile = len(dfs)

    # 各ファイル同士で座標変換による重ね合わせを行い、
    # 最も他のファイルとの一致が良いファイルを後の基準ファイルにする
    scorelist = [[0]*numFile for i in range(numFile)]
    rotateMatrixList = [[0]*numFile for i in range(numFile)]
    transMatrixList = [[0]*numFile for i in range(numFile)]
    for i, j in itertools.combinations(range(numFile), 2):
        # i < j
        df_i = dfs[i]
        df_j = dfs[j]
        refAtomIndexes_i = refAtomIndexesList[i]
        refAtomIndexes_j = refAtomIndexesList[j]

        if len(refAtomIndexes) != 0:
            df_i_refs = df_i.iloc[refAtomIndexes]
            df_j_refs = df_j.iloc[refAtomIndexes]
        else:
            df_i_refs = df_i
            df_j_refs = df_j

        # DataFrameとしてはここでは使わないのでndarrayに変換
        xyz_i_refs = df_i_refs[['x','y','z']].values
        xyz_j_refs = df_j_refs[['x','y','z']].values
        # 変換パラメータ推定
        # Rとtを求める
        R, t, score = estimate_conversionParameter(xyz_i_refs, xyz_j_refs)
        rotateMatrixList[i][j] = R
        transMatrixList[i][j] = t

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
    for j, df_j in enumerate(dfs):
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
    dfs = readFiles(args.file, args.format)
    # 一致させる基準原子のインデックスを取得
    df0refsIndexes = [int(s) for s in args.ref]

    # 各ファイルを一致させるための変換行列を取得する
    if args.sorted==False or args.best==False:
        refAtomIndexes = []
        conversionParameterList = []
        for j in range(1,len(dfs)):
            R, t, dfjrefsIndexes = getConversionParameter(dfs[0], dfs[j], df0refsIndexes, args.sorted)
            refAtomIndexes.append(dfjrefsIndexes)
            conversionParameterList.append([R,t])
    else:
        refAtomIndexes = df0refsIndexes * len(dfs)

    if args.best==True:
        conversionParameterList = getMostSuitableConversionParameterList(dfs, refAtomIndexes)

    # 各ファイルを一致させる(変換)
    # 変換後のデータはリストに記録していき、最後にファイル出力する
    convertedxyzdata = []
    for df_i, (R, t) in zip(dfs,conversionParameterList):
        xyz_i_converted = df_i[['x','y','z']].values @ R.T + t
        convertedxyzdata.extend(['{} {} {} {}'.format(s,x,y,z) for s,(x,y,z) in zip(df_i['elementSymbol'], xyz_i_converted)])

    # 変換結果をxyzファイルに書き出し
    with open(args.save, mode='w') as f:
        f.write('\n'.join(convertedxyzdata))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ovwmols : Overwrite Molecule Files.')

    parser.add_argument('--format', help='molecule file format', required=True, choices=['XYZ','GJF'], type=str)
    parser.add_argument('--ref', help='conversion reference atoms(0-based)', default=[], nargs='*')
    parser.add_argument('--file', help='molecule file', required=True, nargs='*')
    parser.add_argument('--sorted', help='molecule file is sorted already', action='store_true')
    parser.add_argument('--best', help='convert to best fit', action='store_true')
    parser.add_argument('--save', help='save destination', default='ovwmols.xyz', type=str)

    # 引数解析
    # main実行
    main(parser.parse_args())


