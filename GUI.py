from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
import subprocess
import time

def bb_():
    pop3 = Toplevel(root)
    pop3.transient(root)
    pop3.geometry("200x100")
    pop3.title("版本")
    labelp3 = Label(pop3, text="目前版本号：\n  v 1.0", font=("Arial", 16))
    labelp3.pack()

def dj_():
    pop1 = Toplevel(root)
    pop1.transient(root)
    pop1.geometry("300x150")
    pop1.title("选择")
    labelp1 = Label(pop1, text="请选择AI水平", font=("Arial", 14))
    labelp1.pack()
    rb1 = Radiobutton(pop1, text="初级", variable=std_, value=1)
    rb1.place(x=20, y=30)
    rb2 = Radiobutton(pop1, text="中级", variable=std_, value=2)
    rb2.place(x=100, y=30)
    rb3 = Radiobutton(pop1, text="高级", variable=std_, value=3)
    rb3.place(x=180, y=30)
    b3 = Button(pop1, text='确定', height=1, width=4, command=pop1.destroy, font=("Arial", 18)
                , relief='raised').place(x=100, y=80)

def bg_():
    global img_bg2
    global img_bg_2
    img_bg2 = Image.open('bg2.jpg')
    img_bg_2 = ImageTk.PhotoImage(img_bg2)
    canvas1.configure(image=img_bg_2)
    canvas1.place(x=0, y=0)

def cx_():
    sys.exit()

def save_():
    file = open('data_save.txt', 'w')
    file.write(his_v.get())

def fx1(a):
    aa1 = ''
    if a =='PASS':
        return 'PASS'
    else:
        a1 = a.split()
        if a1[0] == 'PLAY':
            aa1 = '出牌：'+fx2(a1[1])

        elif a1[0] == 'CHI':
            aa1 = '吃：'+fx2(a1[1])+",出牌"+fx2(a1[2])
        elif a1[0] == 'PENG':
            aa1 = '碰：'+fx2(a1[1])
        elif a1[0] == 'GANG':
            aa1 = '杠 ：'+fx2(a1[1])
        elif a1[0] == 'ANGANG':
            aa1 = '暗杠 ：'+fx2(a1[1])
        elif a1[0] == 'BUGANG':
            aa1 = '补杠 ：'+fx2(a1[1])
        return  aa1

def fx2(a):
    aa2 = 'NO'
    if a[0] == 'B':
        aa2 = a[1]+'饼'
    elif a[0] == 'T':
        aa2 = a[1] + '条'
    elif a[0] == 'W':
        aa2 = a[1] + '万'
    elif a[0] == 'F':
        if a[1] == "1":
            aa2 = '东风'
        elif a[1] == "2":
            aa2 = '南风'
        elif a[1] == "3":
            aa2 = '西风'
        elif a[1] == "4":
            aa2 = '北风'
    elif a[0] == 'J':
        if a[1] == "1":
            aa2 = '中'
        elif a[1] == "2":
            aa2 = '发'
        elif a[1] == "3":
            aa2 = '白板'
    return aa2

def shoupai():

    img_f = Image.open(sp_[-1] + '.png')
    tk_img = ImageTk.PhotoImage(img_f)
    img_.append(tk_img)
    for i2,j in enumerate(sp_):
        Label(frame_top, image=img_[i2]).place(x=40+40*i2,y=50)

def sp_set(a):
    c_[0] = a
    c_v.set(a)
    if flag[0] ==1:
        if len(sp_)<13:
            sp_.append(a)
            for w in frame_top.winfo_children()[1:]:
                w.destroy()
            shoupai()
    else:
        hp_[0]= a
        img_f3 = Image.open(a + '.png')
        img_[0] = ImageTk.PhotoImage(img_f3)
        Label(frame8, image=img_[0]).place(x=40,y=120)

def qd():   # 选择玩家后
    if player.get()<5 and player.get() >0:
        pl[0]= player.get()
        root.winfo_children()[-1].destroy()
    j = ""
    for i in sp_:
        j += " "+i
    AI.stdin.write('0\n'.encode())
    AI.stdin.flush()
    AI.stdin.write(str('0 '+str(int(pl[0])-1)+' 0\n').encode())
    AI.stdin.flush()
    AI.stdin.write(str('1 0 0 0 0'+j+'\n').encode())
    AI.stdin.flush()
    print(AI.stdout.read1().decode())
    print(AI.stdout.read1().decode())
    print(AI.stdout.read1().decode())
    print(AI.stdout.read1().decode())
    new()

def ks():
    if len(sp_)<13:
        pop = Toplevel(root)
        pop.geometry("300x200")
        pop.title("错误")
        labelp = Label(pop, text="\n\n\n请输入13张手牌!", font=("Arial", 18))
        labelp.pack()
    else:
        flag[0]=2
        for widget in frame_top.winfo_children():
            widget.destroy()
        for widget in frame_.winfo_children():
            widget.destroy()
        for widget in frame_left.winfo_children():
            widget.destroy()

        pop = Toplevel(root)
        pop.geometry("300x150")
        pop.title("选择")
        labelp = Label(pop, text="请选择你的位置", font=("Arial", 14))
        labelp.pack()
        rb1 = Radiobutton(pop,text="玩家1",variable=player,value=1)
        rb1.place(x=20,y=30)
        rb2 = Radiobutton(pop, text="玩家2", variable=player, value=2)
        rb2.place(x=80,y=30)
        rb3 = Radiobutton(pop, text="玩家3", variable=player, value=3)
        rb3.place(x=140,y=30)
        rb4 = Radiobutton(pop, text="玩家4", variable=player, value=4)
        rb4.place(x=200,y=30)
        b3 = Button(pop, text='确定', height=1, width=4, command=qd, font=("Arial", 18)
                    , relief='raised').place(x=100,y=80)

    return 0

def new():
    global txt
    global frame8
    global frame10
    frame5 = Frame(frame_left,bd= 3,height= 200,width=200,relief= 'ridge')
    frame5.place(x=0, y=0)
    frame7.append(Frame(frame_left,bd= 3,height= 300,width=200,relief= 'ridge'))
    frame7[0].place(x=0, y=200)
    label3 = Label(frame7[0], text="选择动作", font=("Arial", 14))
    label3.place(x=0, y=0)
    p1 = Button(frame5, text='玩家1', height=1, width=6, command=lambda a='1':dz1(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=20)
    p2 = Button(frame5, text='玩家2', height=1, width=6, command=lambda a='2':dz1(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=60)
    p3 = Button(frame5, text='玩家3', height=1, width=6, command=lambda a='3':dz1(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=100)
    p4 = Button(frame5, text='玩家4', height=1, width=6, command=lambda a='4':dz1(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=140)
    aa = pl[0]-1
    frame5.winfo_children()[aa].destroy()
    p5 = Button(frame5, text='我', height=1, width=6, command=lambda a=str(pl[0]):dz2(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=20+40*aa)
    p6 = Button(frame, text='确定', height=1, width=6, command=que, font=("Arial", 14)
                , relief='raised').place(x=700, y=450)

    frame8 = Frame(frame,bd= 3,height= 200,width=150,relief= 'ridge',bg="lightblue")
    frame8.place(x=200, y=0)
    Label(frame8,text="输入状态:",font=("Arial",18) ,bg="lightblue").place(x=10, y=10)
    Label(frame8, textvariable=all_v, font=("Arial", 18),bg="lightblue").place(x=10, y=60)

    frame9 = Frame(frame, bd=3, height=200, width=150, relief='ridge',bg="lightyellow")
    frame9.place(x=350, y=0)
    Label(frame9, text="AI推荐", font=("Arial", 18),bg="lightyellow").place(x=10, y=10)
    Label(frame9, textvariable=tuijian_v, font=("Arial", 18),bg="lightyellow").place(x=10, y=60)

    frame10 = Frame(frame, bd=3, height=200, width=300, relief='ridge')
    frame10.place(x=500, y=0)
    Label(frame10, text="推荐历史", font=("Arial", 18)).place(x=80, y=5)
    txt = Text(frame10,height=6, width=20,state=DISABLED, font=("Arial", 18),bg="lightgreen")
    txt.place(x=0, y=35)
    txt.tag_config('tag', justify='left',font=("Arial", 12))
    yscroll = Scrollbar(frame10, orient=VERTICAL, command=txt.yview)
    yscroll.place(x=278,y=35,height=178)
    txt['yscrollcommand'] = yscroll.set

def act(a):
    a_[0] = dz_e[a]
    a_v.set(dz_h[a])
    all_v.set(p_v.get() + a_v.get()+":")
    print("嗷呜呜",type(p_[0]),type(a_[0]),type(c_[0]))

def que():
    if p_[0]!=0 and a_[0]!=0 and c_[0]!=0:

        if a_[0] ==' DRAW ':
            act_1 = '2 '+c_[0]
        else:
            act_1 = '3 '+str(int(p_[0][-1])-1)+a_[0]+c_[0]
        AI.stdin.write(str(act_1+'\n').encode())
        AI.stdin.flush()
        tj_ = AI.stdout.read1().decode()
        tj_ = fx1(tj_)
        tuijian_v.set(tj_)
        print(AI.stdout.read1().decode())
        print(tuijian_v.get())
        txt['state'] = NORMAL
        txt.insert('end', p_v.get() +" "+ a_v.get()+c_v.get() + '\nAI推荐    '+tuijian_v.get()+'\n\n', 'tag')
        txt['state'] = DISABLED

def dz1(a):
    p_[0] = 'Player '+ a
    p_v.set("玩家"+str(a))
    all_v.set(p_v.get() + a_v.get()+":")

    if frame7[0].winfo_children()[1:]:
        for w2 in frame7[0].winfo_children()[1:]:
            w2.destroy()
    d21 = Button(frame7[0], text='出牌', height=1, width=6, command=lambda a=1:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=60)
    d31 = Button(frame7[0], text='吃', height=1, width=6, command=lambda a=2:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=100)
    d41 = Button(frame7[0], text='碰', height=1, width=6, command=lambda a=3:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=160)
    d51 = Button(frame7[0], text='杠', height=1, width=6, command=lambda a=4:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=200)
    label51 = Label(frame7[0], text='注：吃牌需要选3张牌中间那张', font=("Arial", 10)
                 , relief='raised').place(x=1, y=260)

def dz2(a):
    p_[0] = 'Player ' + a
    p_v.set("玩家"+str(a))
    all_v.set(p_v.get() + a_v.get()+":")

    if frame7[0].winfo_children()[1:]:
        for w2 in frame7[0].winfo_children()[1:]:
            w2.destroy()
    d12 = Button(frame7[0], text='抽牌', height=1, width=6, command=lambda a=0:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=40)
    d22 = Button(frame7[0], text='出牌', height=1, width=6, command=lambda a=1:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=80)
    d32 = Button(frame7[0], text='吃', height=1, width=6, command=lambda a=2:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=120)
    d42 = Button(frame7[0], text='碰', height=1, width=6, command=lambda a=3:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=160)
    d52 = Button(frame7[0], text='杠', height=1, width=6, command=lambda a=4:act(a), font=("Arial", 14)
                , relief='raised').place(x=60, y=200)
    label52 = Label(frame7[0], text='注：吃牌需要选3张牌中间那张', font=("Arial", 10)
                 , relief='raised').place(x=1, y=260)

def ht():   #回退
    if len(sp_)>0:
        sp_.pop()
    for widget in frame_top.winfo_children()[1:]:
        widget.destroy()
    shoupai()


AI = subprocess.Popen(['python', '__main__.py'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)

txt=1
root = Tk()
root.geometry("800x500")
frame = Frame(root)
frame.place(x=0,y=0)
menu = Menu(root)
menue_1 = Menu(menu,tearoff=0)
menue_2 = Menu(menu,tearoff=0)
menue_3 = Menu(menu,tearoff=0)
menu.add_cascade(label="菜单",menu=menue_1)
menu.add_cascade(label="文件",menu=menue_2)
menu.add_cascade(label="设置",menu=menue_3)
menue_1.add_command(label="退出",command=sys.exit)
menue_1.add_command(label="下一局",command=cx_)
menue_2.add_command(label="保存对局",command=save_)
menue_2.add_command(label="加载对局")
menue_3.add_command(label="切换背景",command=bg_)
menue_3.add_command(label="切换等级",command=dj_)
menue_3.add_command(label="显示版本",command=bb_)
root.config(menu=menu)

img_bg = Image.open('bg1.jpg')
img_bg_ = ImageTk.PhotoImage(img_bg)
canvas1 = Label(frame, width=800, height=500, image=img_bg_)
canvas1.place(x=0, y=0)

player = IntVar()
text_ = StringVar()

pl = [0]  # 玩家
sp_ = []  # 初始手牌
hp_ = [0]  # 显示牌
dz_e = [' DRAW ', ' PLAY ', ' CHI ', ' PENG ', ' GANG ']
dz_h = ['抽牌 ', '出牌 ', '吃 ', '碰 ', '杠 ']
p_ = [0]  # 输出玩家
a_ = [0]  # 输出动作
c_ = [0]  # 输出牌
p_v = StringVar()
a_v = StringVar()
c_v = StringVar()
all_v = StringVar()
his_v = StringVar()
frame7 = []
flag = [1]  # 点开始的标志位
tuijian_v = StringVar()

frame_left = Frame(frame, width=200, height=500, relief='ridge')
frame_left.pack(side=LEFT)

frame_top = Frame(frame, width=600, height=150, relief='ridge')
frame_top.pack(side=TOP)

label1 = Label(frame_top, text="初始手牌", font=("Arial", 18))
label1.place(x=238, y=10)
img_ = []  # 手牌图像

# 选牌部分

frame_ = Frame(frame, height=60, relief='ridge')
frame_.pack(side=TOP)
frame1 = Frame(frame, height=100, relief='ridge')
frame1.pack(side=TOP)
b_1 = []

for i in range(9):
    img_f = Image.open('B' + str(i + 1) + '.png')
    tk_img = ImageTk.PhotoImage(img_f)
    b_1.append(tk_img)

    Button(frame1, image=b_1[i], command=lambda i=i: sp_set('B' + str(i + 1))
           , relief='raised', justify=LEFT).pack(side=LEFT)

b_2 = []
frame2 = Frame(frame, height=100, relief='ridge')
frame2.pack(side=TOP)
for i in range(9):
    img_f = Image.open('T' + str(i + 1) + '.png')
    tk_img = ImageTk.PhotoImage(img_f)
    b_2.append(tk_img)

    Button(frame2, image=b_2[i], command=lambda i=i: sp_set('T' + str(i + 1))
           , relief='raised', justify=LEFT).pack(side=LEFT)

b_3 = []
frame3 = Frame(frame, height=100, relief='ridge')
frame3.pack(side=TOP)
for i in range(9):
    img_f = Image.open('W' + str(i + 1) + '.png')
    tk_img = ImageTk.PhotoImage(img_f)
    b_3.append(tk_img)

    Button(frame3, image=b_3[i], command=lambda i=i: sp_set('W' + str(i + 1))
           , relief='raised', justify=LEFT).pack(side=LEFT)

b_4 = []
b_4_ = []
for i in range(1, 5):
    b_4.append('F' + str(i))
for i in range(1, 4):
    b_4.append('J' + str(i))

frame4 = Frame(frame, height=100, relief='ridge')
frame4.pack(side=TOP)
for i, j in enumerate(b_4):
    img_f = Image.open(j + '.png')
    tk_img = ImageTk.PhotoImage(img_f)
    b_4_.append(tk_img)
    Button(frame4, image=b_4_[i], command=lambda i=j: sp_set(i)
           , relief='raised', justify=LEFT).pack(side=LEFT)

frame5 = Frame(frame_left, bd=3, height=300, width=200, relief='ridge')
frame5.place(x=0, y=0)

label2 = Label(frame5, text="说明：\n\n\n\n   请选择13张初始手\n牌，然后点击开始按\n钮。", font=("Arial", 16), justify=LEFT)
label2.place(x=0, y=20)

frame6 = Frame(frame_left, bd=3, height=200, width=200, relief='ridge')
frame6.place(x=0, y=300)
b1 = Button(frame6, text='开始', height=2, width=10, command=ks, font=("Arial", 18)
            , relief='raised').place(x=30, y=20)
b2 = Button(frame6, text='回退', height=2, width=10, command=ht, font=("Arial", 18)
            , relief='raised').place(x=30, y=100)

p11 = Button(frame, text='退出', height=1, width=6, command=sys.exit, font=("Arial", 14)
            , relief='raised').place(x=200, y=450)

root.title("麻将AI")

std_ = IntVar()
pop1 = Toplevel(root)
pop1.transient(root)
pop1.geometry("300x150")
pop1.title("选择")
labelp1 = Label(pop1, text="请选择AI水平", font=("Arial", 14))
labelp1.pack()
rb1 = Radiobutton(pop1, text="初级", variable=std_, value=1)
rb1.place(x=20, y=30)
rb2 = Radiobutton(pop1, text="中级", variable=std_, value=2)
rb2.place(x=100, y=30)
rb3 = Radiobutton(pop1, text="高级", variable=std_, value=3)
rb3.place(x=180, y=30)

b3 = Button(pop1, text='确定', height=1, width=4, command=pop1.destroy, font=("Arial", 18)
            , relief='raised').place(x=100, y=80)

root.mainloop()







