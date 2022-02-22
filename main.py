import image_cal as im
import graph_cal as gr
import text_cal as tx
import draw_graph

##### 비교할 건물쌍 이름들.
name1 = 'Canter'
name2 = 'Jewish'

subjects = ['Topology', 'Exterior Image(SSIM)', 'Exterior Image(MSE)', 'Interior Image(SSIM)', 'Interior Image(MSE)', 'Concept(Text)',\
                'Plan Image(SSIM)', 'Plan Image(MSE)']
result_list = []

result_list.append(gr.f_score)
result_list.append(im.es1)
result_list.append(im.es2)
result_list.append(im.is1)
result_list.append(im.is2)
result_list.append(tx.f_score)
result_list.append(im.ps1)
result_list.append(im.ps2)
result_list.append(gr.f_score)

print(result_list)

draw_graph.drawG(result_list, name1, name2)
