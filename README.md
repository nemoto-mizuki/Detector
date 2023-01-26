# Detector


### 記録

date 2022/10/09

batch_dict内のデータ構造の調整
batrch_dict['points']の構造を、[time, N, C_in(6)]にしたいが、Nが一定でないため、このデータ構造では不可能
対策案として、dict型に変更する、or N_maxを次元数としてパディングする？(未実装)

#### 修正箇所
- waymo_dataset.py
- waymo_dataset_multiframe.yaml

SPLIT_SEQUENCE: True を追加

data_dict[x, y, z, intensity, elongation, time_split_number]

points = {'{}'.format(i):p[p[:,5]==i] for i in range(4)}  # 4:sequence length

line376 'split_data': True

- dadaset.py line183
            points = {'{}'.format(i):points[points[:,5]==i] for i in range(abs(max(self.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET, key=abs))+1)}


#### 計画(2022/11.10)
backbone2d を、conv3dに変更
CenterHeadに対し、
self-attention moduleの挿入
shared_convを、512->64から変更

  

### メモ
2023/01/26 gitのバグもしくはエラーにより、リポジトリを再構成