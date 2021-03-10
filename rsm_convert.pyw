'''逆格子マッピングデータからグラフを作成'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import re

import matplotlib.font_manager as fmanager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


###################################################################
def setup_logging(logging_level):
    '''initialize logging'''
    log_handler = logging.StreamHandler()
    format = logging.Formatter('%(asctime)s line %(lineno)s in %(name)s | %(message)s')
    log_handler.setFormatter(format)

    logger = logging.getLogger(os.path.basename(__file__))
    logger.addHandler(log_handler)
    logger.setLevel(logging_level)

    return logger


LOGGER = setup_logging(logging.DEBUG)

# ngraphっぽいデザインにする
fmanager._rebuild() #pylint: disable=protected-access
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['xtick.minor.width'] = 2.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['ytick.minor.width'] = 2.0
###################################################################


###################################################################
#                         設定するパラメータ                       #
###################################################################
_FILE = 'sample.txt'                       # ファイル名(このプログラムを同じディレクトリに配置すること!)
LABEL_X = r'$Q_x$ || BFO ($110$) (nm$^{-1}$)' # X軸の名前
LABEL_Y = r'$Q_y$ || BFO ($001$) (nm$^{-1}$)' # Y軸の名前


_USE_DEFAULT = False    # 軸を指定しないときはこれをTrueにする
_FILL = False           # Trueにすると等高線をカラーマップで塗りつぶす
_X_MIN = -0.39          # X軸の最小値
_X_MAX = -0.33          # X軸の最大値
_Y_MIN = 0.72           # Y軸の最小値
_Y_MAX = 0.78           # Y軸の最大値
_AXIS_STEP = 0.02       # 軸間隔


CMAP_TYPE = 'gnuplot'   # 使うカラーマップ (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
CMAP_MAX = 1000         # 1000以上はエラー
FONT_SIZE = 20          # フォントサイズ
###################################################################


def get_file():
    '''パス取得'''
    _chdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(_chdir, _FILE), os.path.splitext(_FILE)[1]


def str2float(string) -> float:
    '''stringはfloatに変換可能か調べる。可能ならfloatにして出力'''
    p_word = r'[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?'
    return float(string) if re.fullmatch(p_word, string) else 0.


# pylint: disable=unsubscriptable-object
def rsm_read(fpath, delimiter='\t') -> np.ndarray:
    '''
    rsmデータ→np.ndarray
    np.loadtxtではダメだった
    '''
    min_max = [[0,0], [0,0]]
    with open(fpath, 'r') as _f:
        LOGGER.debug(f'Load success ({fpath})')
        data = _f.readlines()
        arr_data = []
        for _data in data:
            if 'min' in _data or 'max' in _data:
                ap_data = _data[:-2].split(' ')[3:]
                ap_data = [str2float(x) for x in ap_data]
                min_max[int('max' in _data)] = ap_data
            elif _data == '\n' or _data[0] == '#':
                pass
            else:
                ap_data = _data[:-1].split(delimiter)[:-1]
                ap_data = [str2float(x) for x in ap_data]
                arr_data.append([ap_data])

    min_max = np.array(min_max).T
    if np.any(min_max==0):
        LOGGER.warning('\033[31mCannot read min/max value.\033[0m')
    else:
        LOGGER.debug(f'X axis : [min, max] = {min_max[0]}')
        LOGGER.debug(f'Y axis : [min, max] = {min_max[1]}')

    return np.array(arr_data).T


def make_contour():
    '''メイン'''
    fpath, ext = get_file()
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)

    data = rsm_read(fpath, delimiter='\t')
    q_x = np.unique(data[0][0])
    q_y = np.unique(data[1][0])
    intensity = data[2][0].reshape([len(q_y), len(q_x)])

    fig, ax = plt.subplots(figsize=(5.12,5.12)) #pylint: disable=invalid-name

    attrname = 'contour'
    if _FILL:
        attrname += 'f'
    LOGGER.debug(f'Contour = ax.{attrname}')
    getattr(ax, attrname)(q_x, q_y, intensity, CMAP_MAX, cmap=CMAP_TYPE)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    if not _USE_DEFAULT:
        if (_X_MIN >= _X_MAX or _Y_MIN >= _Y_MAX or _AXIS_STEP <= 0) or \
                abs(_X_MAX-_X_MIN)<_AXIS_STEP or abs(_Y_MAX-_Y_MIN)<_AXIS_STEP:
            LOGGER.warning('\033[31m(WARNING) Invali axis parameters. Disabled.\033[0m')
        else:
            bias = 10**(round(math.log10(_AXIS_STEP))-1)
            axis_params = [
                [_X_MIN, _X_MAX+bias, _AXIS_STEP, bias],
                [_Y_MIN, _Y_MAX+bias, _AXIS_STEP, bias]]
            ax.set_xlim(axis_params[0][0], axis_params[0][1]-axis_params[0][3])
            ax.set_ylim(axis_params[1][0], axis_params[1][1]-axis_params[1][3])
            ax.set_xticks(np.arange(*axis_params[0][:-1]))
            ax.set_yticks(np.arange(*axis_params[1][:-1]))

    ax.set_xlabel(LABEL_X)
    ax.set_ylabel(LABEL_Y)

    plt.subplots_adjust(left=0.2,bottom=0.2)
    fig.savefig(fpath.replace(ext, '.png'), dpi=400)

    LOGGER.debug('Image saved to "'+fpath.replace(ext, '.png')+'"')
    LOGGER.debug('Display figures')

    plt.show()
    plt.close()

    LOGGER.debug('End')
# pylint: enable=unsubscriptable-object


if __name__ == '__main__':
    LOGGER.debug('Start')
    if CMAP_MAX > 1000:
        CMAP_MAX = 1000
    elif CMAP_MAX < 10:
        CMAP_MAX = 10
    make_contour()
