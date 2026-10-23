"""Assemble evidence from completed experiments; never invent missing metrics."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import json
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import HERE,save_json


def evidence_inventory():
    """File presence is evidence availability, never a scientific acceptance gate."""
    requirements={
        'S6_ten_layer_fields':['layers_n10_total30_os2/evaluation/S6_plane_fields.npz',
                               'layers_n10_total30_os2/evaluation/S6_longitudinal.npz'],
        'S7A_layers':[f'layers_n{n}_total30_os2/sweep_metrics.json' for n in [0,1,2,3,4,5,10]],
        'S7A_gaps':[f'gap_{n}mm_os2/sweep_metrics.json' for n in [1,2,3,5,6,7,8,9,10]]+['baseline_os2/sweep_metrics.json'],
        '1550nm_S7A_layers':[f'layers_n{n}_total30_1550nm_os16/sweep_metrics.json' for n in [0,1,2,3,4,5,10]],
        '1550nm_S7A_gaps':[f'gap_{n}mm_1550nm_os16/sweep_metrics.json' for n in [1,2,3,5,6,7,8,9,10]]+['wavelength1550_paper_geometry_os16/sweep_metrics.json'],
        '1550nm_S6_ten_layer_fields':['layers_n10_total30_1550nm_os16/evaluation/S6_plane_fields.npz',
            'layers_n10_total30_1550nm_os16/evaluation/S6_longitudinal.npz'],
        'S7B_natural_images':['baseline_os2/evaluation/test_metrics.json','baseline_os2/evaluation/per_image.csv'],
        'S7C_resolution':['baseline_os2/evaluation/resolution_chart_both_axes.npz',
                           'baseline_os2/evaluation/synthetic_metrics_selected_patterns.json'],
        'S7C_combined_errors':['baseline_os2/evaluation/full_error_metrics.json'],
        'S7C_error_mechanisms':['baseline_os2/evaluation/error_ablation_validation.json'],
        'S8_numerical_analogues':['baseline_os2/evaluation/synthetic_metrics.json'],
        'S8_illumination_scan':['baseline_os2/evaluation/S8_measurement_conditions_v2/measurement_metrics.json'],
        'S8_position_mapping':['baseline_os2/evaluation/S8_measurement_conditions_v2/position_analysis.json'],
        'numerical_convergence':['baseline_os2/evaluation/numerical_convergence.json'],
        'error_sampling_convergence':['baseline_os2/evaluation/error_numerical_convergence.json'],
        'larger_training_sample_p10':['imagenet_train10000_1550nm_os16/training_complete.json',
            'imagenet_train10000_1550nm_os16/evaluation/test_metrics.json','data_size_comparison.json'],
        'laboratory_1550nm_control':['wavelength1550_paper_geometry_os16/training_complete.json',
            'wavelength1550_paper_geometry_os16/evaluation/test_metrics.json',
            'wavelength1550_paper_geometry_os16/evaluation/resolution_chart_both_axes.png',
            'wavelength1550_paper_geometry_os16/evaluation/numerical_convergence.json',
            'wavelength1550_paper_geometry_os16/evaluation/high_resolution_validation.json',
            'wavelength1550_paper_geometry_os16/evaluation/high_resolution_validation.png',
            'wavelength1550_paper_geometry_os16/evaluation/S8_measurement_conditions_v2/measurement_metrics.json',
            'wavelength1550_paper_geometry_os16/evaluation/S8_measurement_conditions_v2/position_analysis.json'],
    }
    return {name:{'available':[p for p in paths if (HERE/'results'/p).is_file()],
                  'missing':[p for p in paths if not (HERE/'results'/p).is_file()],
                  'scientific_acceptance':'not_established_by_file_presence'}
            for name,paths in requirements.items()}


def validation_diagnostics():
    rows=[]
    names=['baseline_os2','baseline_os1_diagnostic','uniform_init_pilot10',
           'intensity_gray_pilot10','lr010_init01_pilot10','lr001_init07_pilot10','uniform_init_full50','lr010_init01_full50']
    for name in names:
        run=HERE/'results'/name
        if not (run/'training_history.csv').exists():continue
        config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        with (run/'training_history.csv').open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f))
        if not history:continue
        best=min(history,key=lambda row:float(row['val_mse']))
        rows.append({'run':name,'epochs_observed':int(history[-1]['epoch']),
                     'epochs_requested':config['epochs'],'selected_epoch':int(best['epoch']),
                     'best_validation_mse':float(best['val_mse']),
                     'validation_amplitude_ssim_at_best_mse':float(best['val_ssim_amplitude']),
                     'input_encoding':config.get('input_encoding','amplitude_gray'),
                     'oversample':config['geometry']['oversample'],
                     'completion_marker_present':(run/'training_complete.json').is_file()})
    return rows


def validation_curves():
    """Compare matched main-imager settings; show unfinished curves as observed."""
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['baseline_os2','uniform_init_full50','lr010_init01_full50']:
        root=HERE/'results'/name
        if not (root/'training_history.csv').exists():continue
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        with (root/'training_history.csv').open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f))
        if not history:continue
        if config.get('input_encoding','amplitude_gray')!='amplitude_gray' or config['geometry']['oversample']!=2:
            raise ValueError('Validation-curve comparison requires the same input encoding and sampling')
        label=f"init {config['geometry']['init_std']:g}, LR {config['lr']:g} ({len(history)}/50)"
        epochs=[int(r['epoch']) for r in history]
        for ax,key in zip(axes,['val_mse','val_ssim_amplitude']):
            values=[float(r[key]) for r in history]
            line,=ax.plot(epochs,values,label=label,linewidth=1.5)
            ax.plot(epochs[-1],values[-1],'o',markersize=4,color=line.get_color())
    axes[0].set_yscale('log');axes[0].set_ylabel('Validation raw intensity MSE')
    axes[1].set_ylabel('Validation amplitude SSIM')
    for ax in axes:
        ax.set_xlim(1,50);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fixed data / 5 masks / 27 mm / OS2; partial runs end at observed epoch')
    fig.tight_layout();fig.savefig(HERE/'results'/'validation_parameter_curves.png',dpi=150);plt.close(fig)


def architecture_curves(rows,path,wavelength_label):
    layer=sorted([r for r in rows if r['run'].startswith('layers_')],key=lambda r:r['layers'])
    gaps=sorted([r for r in rows if r['run'].startswith('gap_') or r['run'] in {'baseline_os2','wavelength1550_paper_geometry_os16'}],key=lambda r:r['gap_mm'])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,rr,key,title in [(axes[0],layer,'layers','Total distance 30 mm (allocation assumption)'),(axes[1],gaps,'gap_mm','Five masks; input 4 mm, output 7 mm')]:
        expected=[0,1,2,3,4,5,10] if key=='layers' else list(range(1,11))
        observed={round(r[key],6) for r in rr}
        pending=[value for value in expected if value not in observed]
        ax.plot([r[key] for r in rr],[r['test100_amp_ssim'] for r in rr],'o-',label='Amplitude SSIM')
        ax.plot([r[key] for r in rr],[r['test100_intensity_ssim'] for r in rr],'s--',label='Intensity SSIM')
        ax.set_xticks(expected);ax.set_xlim(min(expected)-.3,max(expected)+.3)
        status=f'{len(expected)-len(pending)}/{len(expected)} settings evaluated'
        ax.set_xlabel(key);ax.set_ylabel('Mean SSIM, fixed 100 images');ax.set_title(title+'\n'+status);ax.legend()
        if pending:
            ax.text(.02,.02,'Pending: '+', '.join(map(str,pending)),transform=ax.transAxes,fontsize=8)
    fig.suptitle(wavelength_label)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    validation_curves()
    rows=[]
    for p in sorted((HERE/'results').glob('*/sweep_metrics.json')):
        j=json.loads(p.read_text(encoding='utf-8'));g=j['geometry']
        rows.append({'run':p.parent.name,'layers':g['layers'],'gap_mm':g['interlayer_mm'],
                     'wavelength_nm':299792458/g['frequency_hz']*1e9,
                     'total_mm':g['input_mm']+g['output_mm']+(g['layers']-1)*g['interlayer_mm'],
                     'test100_amp_ssim':j['test100']['ssim_amplitude'],
                     'test100_intensity_ssim':j['test100']['ssim_intensity'],
                     'validation_mse':j['validation']['mse']})
    paper_rows=[r for r in rows if math.isclose(r['wavelength_nm'],299792458/.4e12*1e9,rel_tol=1e-9)]
    infrared_rows=[r for r in rows if math.isclose(r['wavelength_nm'],1550,rel_tol=1e-9)]
    architecture_curves(paper_rows,HERE/'results'/'S7A_architecture_curves.png','Paper reference: 0.4 THz')
    if infrared_rows:
        architecture_curves(infrared_rows,HERE/'results'/'S7A_architecture_curves_1550nm.png','1550 nm adaptation; separate from paper reference')
    baseline=HERE/'results'/'baseline_os2'
    metrics=json.loads((baseline/'evaluation'/'test_metrics.json').read_text())
    numerical=json.loads((baseline/'evaluation'/'numerical_convergence.json').read_text())
    error=json.loads((baseline/'evaluation'/'full_error_metrics.json').read_text())
    inventory=evidence_inventory();diagnostics=validation_diagnostics()
    text=['# 全光成像复现结果（自动汇总，仍需物理与图像验收）','',
          '原文和未给参数详见 paper_alignment.md；图像均为数值仿真。','',
          '用户实验波长现为1550 nm；当前主实验为 [OS16成像模型](wavelength1550_paper_geometry_os16/)。',
          '输入距离保持4 mm；下列baseline及论文结构扫描表属于已保留的0.4 THz参考实验。',
          '1550 nm的OS2实验因采样误差停止，高分不能作为成像结论；详见 [固定权重采样核查](wavelength1550_paper_geometry50/evaluation/numerical_convergence.json)。','',
          f"初始基线baseline_os2，300张测试：幅度SSIM={metrics['network']['ssim_amplitude']:.5f}，强度SSIM={metrics['network']['ssim_intensity']:.5f}。",
          f"同距离自由传播幅度SSIM={metrics['free_space']['ssim_amplitude']:.5f}。",
          f"100张固定测试幅度SSIM={metrics['ssim_amplitude_mean_first100']:.5f}。",
          f"Poisson+吸收+失准幅度SSIM（三次平均）={error['mean_ssim_amplitude']:.5f}。",'',
          '|实验|层数|间距mm|总距离mm|验证MSE|100张幅度SSIM|','|---|---:|---:|---:|---:|---:|']
    for r in paper_rows:text.append(f"|{r['run']}|{r['layers']}|{r['gap_mm']:.4g}|{r['total_mm']:.4g}|{r['validation_mse']:.6g}|{r['test100_amp_ssim']:.5f}|")
    scan_state=HERE/'results'/'suite1550_state.json'
    if scan_state.exists():
        scan=json.loads(scan_state.read_text(encoding='utf-8'))
        state_labels={'waiting_for_prerequisites':'等待主模型及大样本对照完成',
                      'running':'逐项训练与评估中','requested_queue_completed':'扫描队列已执行完成，仍需科学验收','failed':'队列已停止，需检查错误'}
        text+=['','## 1550 nm剩余扫描执行状态','',
               state_labels.get(scan['status'],scan['status'])+'。',
               f"已完成{len(scan.get('completed',[]))}/{len(scan.get('planned_runs',[]))}组新增训练与配套图像。", 
               '输入保持4 mm；与主模型、大样本训练串行执行，不同时启动多个训练。']
    if infrared_rows:
        text+=['','## 1550 nm层数与间距扫描','',
               '沿用论文的结构扫描项目，但波长为1550 nm；不与0.4 THz点混画。',
               '总距30 mm的分段分配仍是已披露的实现选择；输入保持4 mm，单层输出26 mm。',
               '![1550 nm结构扫描](S7A_architecture_curves_1550nm.png)']
    data_ready=HERE/'data_imagenet_train10000'/'data_ready.json'
    if data_ready.exists():
        prepared=json.loads(data_ready.read_text(encoding='utf-8'))
        counts=prepared['split_counts']
        text+=['','## 扩展数据准备','',
               f"独立数据集已准备：训练{counts['train']}张、验证{counts['validation']}张、测试{counts['test']}张。",
               '原2000张数据和缓存保持不变；原验证/测试身份及顺序保留，全部10500张内容哈希互不重复。',
               f"独立复核记录：{'已完成' if prepared.get('independent_verification') else '待完成'}；[数据校验记录](../data_imagenet_train10000/data_ready.json)。",
               '数据就绪不代表大样本训练或评估已完成。']
        large_status=HERE/'misc'/'large_sample_status.json'
        if large_status.exists():
            state=json.loads(large_status.read_text(encoding='utf-8-sig'))
            status_labels={'waiting_for_prerequisites':'等待主模型训练和评估完成',
                           'training_10000_images':'10000张图像训练正在运行',
                           'evaluating_10000_image_model':'10000张图像模型正在评估'}
            label=('训练和评估流程已执行完成，仍需科学验收' if state.get('exit_code')==0 else
                   '流程失败，需检查日志' if state.get('exit_code') is not None else
                   status_labels.get(state.get('status'),state.get('status','状态未记录')))
            text.append('大样本流程：'+label+'。')
    selection_path=HERE/'results'/'main_model_selection.json'
    high_resolution=HERE/'results'/'wavelength1550_paper_geometry_os16'/'evaluation'/'high_resolution_validation.json'
    if high_resolution.exists():
        audit=json.loads(high_resolution.read_text(encoding='utf-8'))
        text+=['','## 1550 nm独立CPU采样核查','',
               f"固定第{audit['checkpoint_epoch']}轮权重及前两张验证图，比较OS8/16/32，全部保持同一物理像元和几何。",
               f"同网格Torch与独立CPU实现的最大强度差为{audit['torch_vs_scipy_os8_max_absolute_intensity_difference']:.3g}。",
               'OS32仍为有限参考；两张固定验证图的结果不能代表全数据收敛。','',
               '|采样|相对OS32强度L2差|两张验证图的幅度SSIM|','|---:|---:|---|']
        for row in audit['rows']:
            scores=', '.join(f'{value:.5f}' for value in row['validation_amplitude_ssim'])
            text.append(f"|OS{row['oversample']}|{row['relative_intensity_l2_vs_os32']:.5f}|{scores}|")
        text+=['','![固定权重的独立传播核查](wavelength1550_paper_geometry_os16/evaluation/high_resolution_validation.png)',
               '原始输入、相位及输出保存在同目录high_resolution_validation.npz；本节权重身份以记录的轮次和SHA256为准。']
        fine=next((row for row in audit['rows'] if row['oversample']==32),None)
        free_scores=audit.get('free_space_os32_amplitude_ssim')
        if fine and free_scores:
            network_mean=sum(fine['validation_amplitude_ssim'])/len(fine['validation_amplitude_ssim'])
            free_mean=sum(free_scores)/len(free_scores)
            text+=['',f'OS32下两张验证图的平均幅度SSIM：网络{network_mean:.5f}，自由传播{free_mean:.5f}。',
                   '不同采样网格的分数仍有变化，不能将当前OS16高分视为完整收敛验收；大样本对照及结构扫描仍属于该已披露网格上的数值实验。']
            if 'free_space_os32_intensity_mse' in audit:
                network_mse=fine['validation_intensity_mse'];free_mse=audit['free_space_os32_intensity_mse']
                text.append(f'同两张图的OS32强度MSE：网络{network_mse:.7g}，自由传播{free_mse:.7g}。')
                if network_mse>free_mse:
                    text.append('这两张图上，OS16同网格测试的MSE优势未在更细网格保持。不能将其直接作为物理成像改善的证据；后续OS16样本规模与结构对照本身也不能消除这一采样限制。')
        extended=audit.get('extended_reference')
        if extended:
            text+=['','### OS48扩展核查','',
                   '保持同两张验证图及相同权重，逐块重新计算传播函数以控制CPU内存；不改变训练、物理像元或几何。',
                   'OS48仍是有限参考，这两张图不能证明全数据集收敛。','',
                   '|采样|相对OS48强度L2差|强度MSE|两张幅度SSIM|','|---:|---:|---:|---|']
            for row in extended['rows']:
                scores=', '.join(f'{value:.5f}' for value in row['validation_amplitude_ssim'])
                text.append(f"|OS{row['oversample']}|{row['relative_intensity_l2_vs_os48']:.5f}|{row['validation_intensity_mse']:.7g}|{scores}|")
            free_scores=', '.join(f'{value:.5f}' for value in extended['free_space_amplitude_ssim'])
            text.append(f"同网格自由传播：强度MSE={extended['free_space_intensity_mse']:.7g}，幅度SSIM={free_scores}。")
    if selection_path.exists():
        selection=json.loads(selection_path.read_text(encoding='utf-8'));chosen=selection['selected']
        text+=['','## 0.4 THz参考：等预算验证选择的主模型','',
               f"三组候选均完成50轮，按验证强度MSE选择 {chosen['run']}，最佳轮次 {chosen['selected_epoch']}。",
               '选择不读取测试指标；初始基线及结构扫描仍单独保留，避免混用不同初始化/学习率。','',
               '|候选|学习率|初始化标准差|最佳验证MSE|','|---|---:|---:|---:|']
        for r in selection['candidates']:
            text.append(f"|{r['run']}|{r['learning_rate']:g}|{r['init_std']:g}|{r['validation']['mse']:.6g}|")
        selected_out=Path(chosen['checkpoint']).parent/'evaluation'
        if (selected_out/'test_metrics.json').exists():
            j=json.loads((selected_out/'test_metrics.json').read_text(encoding='utf-8'))
            text+=['',f"所选模型300张测试幅度SSIM={j['network']['ssim_amplitude']:.5f}，强度SSIM={j['network']['ssim_intensity']:.5f}，强度MSE={j['network']['mse']:.6g}。",
                   f"[所选模型图像与指标]({chosen['run']}/evaluation/test_metrics.json)。"]
        else:text+=['','所选模型的测试评估尚未产生，不能宣称验收完成。']
        selected_synthetic=selected_out/'synthetic_metrics.json'
        if selected_synthetic.exists():
            synthetic_results=json.loads(selected_synthetic.read_text(encoding='utf-8'))
            # Full evaluation and pattern-only evaluation share this wrapper.
            measurements=synthetic_results.get('measurements',synthetic_results)
            fine=measurements.get('resolution_chart_both_axes',{}).get('line_modulations',[])
            fine=[r for r in fine if abs(r['linewidth_mm']-1.2)<1e-6]
            if fine:
                text+=['','所选模型1.2mm条纹调制度（本项目已披露测量方法）：'+
                       '；'.join(f"{r['orientation']}={r['modulation']:.4f}" for r in fine)+'。',
                       f"![所选模型双方向分辨率]({chosen['run']}/evaluation/resolution_chart_both_axes.png)"]
        selected_errors=selected_out/'full_error_metrics.json'
        if selected_errors.exists():
            errors=json.loads(selected_errors.read_text(encoding='utf-8'))
            text+=['',f"所选模型联合误差（三次平均）幅度SSIM={errors['mean_ssim_amplitude']:.5f}。",
                   f"[所选模型误差明细]({chosen['run']}/evaluation/full_error_metrics.json)。"]
        selected_s8=selected_out/'S8_measurement_conditions_v2'/'measurement_metrics.json'
        if selected_s8.exists():
            text+=['',f"[所选模型S8实验条件数值对照]({chosen['run']}/evaluation/S8_measurement_conditions_v2/measurement_metrics.json)。"]
        text+=['','以下分辨率、误差分解和S8详细表仍对应初始基线baseline_os2；不与所选主模型混用。']
    text+=['','## 尚未完成的实验产物','',
           '下表只核对文件是否已产生；文件齐全不代表达到论文成像质量。','',
           '|项目|已有文件数|缺失文件数|','|---|---:|---:|']
    for name,item in inventory.items():text.append(f"|{name}|{len(item['available'])}|{len(item['missing'])}|")
    text+=['','## 验证集敏感性对照','',
           '![同一输入编码与采样下的主模型验证曲线](validation_parameter_curves.png)','',
           '只在相同输入编码、采样及相同训练预算下比较。进行中的最佳值仍可能改变；',
           '不同灰度编码的目标不同，MSE与SSIM不能直接排序。以下均为验证指标。','',
           '|实验|已训练/预算轮数|输入编码|OS|最佳验证MSE|对应幅度SSIM|',
           '|---|---:|---|---:|---:|---:|']
    for r in diagnostics:
        text.append(f"|{r['run']}|{r['epochs_observed']}/{r['epochs_requested']}|{r['input_encoding']}|{r['oversample']}|{r['best_validation_mse']:.6g}|{r['validation_amplitude_ssim_at_best_mse']:.5f}|")
    chart=baseline/'evaluation'/'synthetic_metrics_selected_patterns.json'
    if chart.exists():
        measurements=json.loads(chart.read_text(encoding='utf-8'))['measurements']
        lines=measurements.get('resolution_chart_both_axes',{}).get('line_modulations',[])
        text+=['','## 双方向分辨率实测','',
               '条纹长度平均后，在已知条纹/间隙中心采样；调制度为(条纹均值−间隙均值)/(两者之和)。',
               '此测量是本项目明确选定的分析方法，原文未给相同判定阈值。',
               '完整剖面见 [双方向剖面图](baseline_os2/evaluation/resolution_profiles_both_axes.png)。','',
               '|线宽mm|方向|调制度|','|---:|---|---:|']
        for r in lines:text.append(f"|{r['linewidth_mm']:.1f}|{r['orientation']}|{r['modulation']:.4f}|")
        text+=['','初始基线的1.2mm至少一个方向出现条纹与间隙反转，尚不支持达到原文最细线宽成像。']
    measurement=baseline/'evaluation'/'S8_measurement_conditions_v2'/'measurement_metrics.json'
    if measurement.exists():
        s8=json.loads(measurement.read_text(encoding='utf-8'))
        text+=['','## S8 实验条件的数值对照','',
               '采用补充p4的20mm光斑FWHM与0.6mm扫描间隔；分别测试幅度和强度FWHM定义。',
               '下列数据为理想相位网络的数值结果，未加入本组未建模的实际照明波前、探测孔径及硬件噪声。',
               '同时保存逐段总距与S8图注29.5mm两种自由传播对照，均不拟合输出增益。','',
               '|光斑FWHM定义|针孔|输入离焦mm|幅度FWHM mm|强度FWHM mm|',
               '|---|---|---:|---:|---:|']
        for r in s8['records']:
            if not r['pattern'].startswith('pinhole_'):continue
            amp=r['sampled_amplitude_fwhm_x_mm'];inten=r['sampled_intensity_fwhm_x_mm']
            amp='未取得' if amp is None else f'{amp:.3f}'
            inten='未取得' if inten is None else f'{inten:.3f}'
            text.append(f"|{r['beam_fwhm_convention']}|{r['pattern']}|{r['input_offset_mm']:g}|{amp}|{inten}|")
        text+=['','FWHM采用主峰相连半高区域及线性插值，散斑下的宽度不能单独证明成像或离焦耐受性。']
    positions=baseline/'evaluation'/'S8_measurement_conditions_v2'/'position_analysis.json'
    if positions.exists():
        mapping=json.loads(positions.read_text(encoding='utf-8'))
        text+=['','## 针孔位置映射与能量集中','',
               '对四个针孔位置的全探测面强度重心做线性拟合；输入位置与输出数据均未变换。',
               '重心映射接近1不是成像的充分证据，自由传播同样可能保留重心。','',
               '|光斑定义|模式|x重心斜率|y重心斜率|目标3mm半径内功率比例均值|',
               '|---|---|---:|---:|---:|']
        for r in mapping['centroid_fits']:
            matrix=r['centroid_mapping_matrix']
            text.append(f"|{r['beam']}|{r['mode']}|{matrix[0][0]:.4f}|{matrix[1][1]:.4f}|{r['mean_fraction_power_within_target_3mm']:.4f}|")
    ablation=baseline/'evaluation'/'error_ablation_validation.json'
    if ablation.exists():
        mechanism=json.loads(ablation.read_text(encoding='utf-8'))
        text+=['','## 误差来源分解（200张验证图）','',
               '采用相同模型与输入，保持原始输出尺度；联合及失准案例固定种子42。',
               '材料系数和基底厚度来自补充p3；Poisson重建设置与失准分布为已披露的实现选择。','',
               '|条件|幅度SSIM|强度MSE|收集功率比例|','|---|---:|---:|---:|']
        for r in mechanism['rows']:
            text.append(f"|{r['case']}|{r['ssim_amplitude']:.5f}|{r['mse']:.6g}|{r['capture_ratio']:.5f}|")
        text+=['','吸收后的原始SSIM同时受亮度与结构影响，不据此单独断言结构失效。']
    convergence=baseline/'evaluation'/'error_numerical_convergence.json'
    if convergence.exists():
        sensitivity=json.loads(convergence.read_text(encoding='utf-8'))
        text+=['','## 误差评估的网格敏感性','',
               '固定前8张验证图、物理相位与高度误差；OS4是有限精度参考，不是解析真值。','',
               '|OS|条件|幅度SSIM|收集功率比例|相对OS4强度L2偏差|',
               '|---:|---|---:|---:|---:|']
        for r in sensitivity['rows']:
            text.append(f"|{r['oversample']}|{r['case']}|{r['ssim_amplitude']:.5f}|{r['capture_ratio']:.5f}|{r['intensity_relative_l2_vs_os4']:.5f}|")
    text+=['','## 尚须人工核实的验收项','',
           '- S6纵向光场、S7B测试图与S7C的1.2mm条纹实际可分辨性。',
           '- 原文SSIM及灰度幅度/强度约定未完整给出，分数接近不等于算法定义一致。',
           '- 当前ImageNet抽样来自明确披露的原始分辨率分片，不是作者未公开的原始索引。',
           '- 数值收敛见 baseline_os2/evaluation/numerical_convergence.json；有限OS参考不能当解析真值。',
           '- 层数扫描的逐段间距分配、Poisson参数和失准分布为披露的复现选择。',
           '- S8原图是实体测试，本项目提供相应数值图案，不声称完成制造及太赫兹测量。']
    infrared=HERE/'results'/'wavelength1550_paper_geometry_os16'
    if (infrared/'evaluation'/'test_metrics.json').exists():
        ir=json.loads((infrared/'evaluation'/'test_metrics.json').read_text(encoding='utf-8'))
        ir_config=json.loads((infrared/'config.json').read_text(encoding='utf-8'))
        base_config=json.loads((baseline/'config.json').read_text(encoding='utf-8'))
        numerical_fields={'frequency_hz','oversample','pad_factor'}
        geometry_equal={k:v for k,v in ir_config['geometry'].items() if k not in numerical_fields}=={k:v for k,v in base_config['geometry'].items() if k not in numerical_fields}
        if not geometry_equal or ir_config['manifest_sha256']!=base_config['manifest_sha256']:
            raise ValueError('Wavelength-only comparison requires identical physical geometry and dataset')
        text+=['','## 实验室1550 nm：保持论文几何的波长对照','',
               '输入4 mm、四段层间各4 mm、输出7 mm；300 μm像元、90 mm口径、五层。',
               '1500/200/300同一数据划分，50轮、batch4、原始强度MSE，从头训练。',
               f"本轮训练网格OS{ir_config['geometry']['oversample']}，正式测试评估网格OS{ir['sampling']['oversample']}；粗网格训练分数不能代替细网格结果。",
               '硬件像元及口径尚未确定，本组尺寸是控制变量仿真假设。',
               '短波长下自由传播本身可能已有很高保真度；高SSIM不能单独证明网络带来改善。','',
               '|波长条件|模式|测试幅度SSIM|测试强度MSE|','|---|---|---:|---:|']
        for label,scores in [('论文0.4 THz',metrics),('实验室1550 nm',ir)]:
            for mode,key in [('D2NN','network'),('自由传播','free_space')]:
                r=scores[key];text.append(f"|{label}|{mode}|{r['ssim_amplitude']:.6f}|{r['mse']:.6g}|")
        reduction=100*(1-ir['network']['mse']/ir['free_space']['mse'])
        ssim_delta=ir['network']['ssim_amplitude']-ir['free_space']['ssim_amplitude']
        text+=['',f'本评估网格上网络相对自由传播的MSE降低{reduction:.2f}%，幅度SSIM差值为{ssim_delta:+.6f}。',
               'MSE和SSIM应分别解释；单项改善不能代替与自由传播的完整成像比较。']
        paper_g=base_config['geometry'];infrared_g=ir_config['geometry']
        paper_lambda=299792458/paper_g['frequency_hz'];infrared_lambda=299792458/infrared_g['frequency_hz']
        text+=['','### 换波长后的归一化尺寸','',
               '下表直接由已保存配置计算，使用λ=c/f；论文0.4 THz对应约749.48 μm，正文常取整为0.75 mm。',
               '像元、口径和距离的毫米数保持一致，但它们相对于波长的比值已经改变；本组不属于原太赫兹条件的光学相似缩放。','',
               '|尺寸/波长|论文0.4 THz配置|1550 nm配置|','|---|---:|---:|']
        for label,key,scale in [('物理像元/λ','pitch_m',1),('输入距离/λ','input_mm',1e-3),
                                ('层间距/λ','interlayer_mm',1e-3),('输出距离/λ','output_mm',1e-3)]:
            text.append(f'|{label}|{paper_g[key]*scale/paper_lambda:.5g}|{infrared_g[key]*scale/infrared_lambda:.5g}|')
        text+=['',f'相同物理尺寸对应的尺寸/波长比增加约{paper_lambda/infrared_lambda:.2f}倍。',
               '因此1550 nm的高自由传播SSIM需要单独报告，不能据此声称复现了原太赫兹成像性能；实验室器件尺寸确定后仍需另行匹配。']
        text+=['','[1550 nm自然图像对照](wavelength1550_paper_geometry_os16/evaluation/test_000.png)；',
               '[双方向分辨率](wavelength1550_paper_geometry_os16/evaluation/resolution_chart_both_axes.png)。',
               'S8的两种20 mm光斑定义、0.6 mm扫描、离焦和离轴图及位置分析在该模型的evaluation/S8_measurement_conditions_v2内。',
               '未将太赫兹材料折射率/吸收系数或制造误差直接用于1550 nm。']
    size_comparison=HERE/'results'/'data_size_comparison.json'
    if size_comparison.exists():
        size= json.loads(size_comparison.read_text(encoding='utf-8'))
        text+=['','## 补充p10：扩大训练样本规模','',
               f"本次比较波长为{size.get('wavelength_nm',749481.145):g} nm，具体控制模型为{size['control_run']}。",
               '若采用1550 nm，该比较属于论文方法的波长扩展，不冒称原论文太赫兹条件的10,000张实验。',
               '原文称10,000张图也得到相近表现，但没有给出该次实验的划分及定量等价阈值。',
               '本项目保留原200/300验证与测试图，将实际训练图由1500张增至10000张；',
               '两组均训练50轮，采用相同的验证选择参数。原始2000张数据与缓存保留。','',
               '|实际训练图数|测试幅度SSIM|测试强度SSIM|测试强度MSE|','|---:|---:|---:|---:|']
        for count,key in [(1500,'control_test'),(10000,'large_test')]:
            r=size[key];text.append(f"|{count}|{r['ssim_amplitude']:.5f}|{r['ssim_intensity']:.5f}|{r['mse']:.6g}|")
        text+=['','同一300张测试图的配对差值与bootstrap区间见 [样本规模比较](data_size_comparison.json)。',
               '置信区间包含零不构成等价证明；不自行指定“相近”的验收阈值。']
    (HERE/'results'/'reproduction_report.md').write_text('\n'.join(text),encoding='utf-8')
    save_json(HERE/'results'/'experiment_matrix.json',{'experiments':rows,'numerical_convergence':numerical,
              'evidence_inventory':inventory,'validation_diagnostics':diagnostics,
              'complete_reproduction_verified':False})
    print('REPORT WRITTEN; numerical and visual completion still requires audit',flush=True)


if __name__=='__main__':main()
