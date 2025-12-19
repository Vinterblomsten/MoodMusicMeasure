from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB', 'pyo', 'pygame']
from psychopy import visual, core, event, sound, gui
import numpy as np
import pandas as pd
import os

def getSubjectInfo():
    info = {'FID': 0, 'BlockOrder': 0}
    dlg = gui.DlgFromDict(dictionary=info, title='n-Armed Bandit Experiment')
    if not dlg.OK:
        core.quit()
    return info['FID'], info['BlockOrder']

def getSubjectCharacteristics():
    info = {'Age': 0, 'Gender (F/M/Other)': ' '}
    dlg = gui.DlgFromDict(dictionary=info, title='n-Armed Bandit Experiment')
    if not dlg.OK:
        core.quit()
    return info['Age'], info['Gender (F/M/Other)']

def trial(win: visual.Window, distributions: list, currentAmount: int, ms: event.Mouse):
    bandits = nbandits(win, distributions)
    aggregate = visual.TextStim(win, text=f"Current Amount: {currentAmount}", pos=(0, -300), color='white', height=60)
    aggregate.draw()
    stims = []
    for (stim, _, _) in bandits:
        stims.append(stim)
        stim.draw()
    drawlabels = floating_key_labels(win, stims[0], stims[1], base_offset=40, label_height=36)
    maxwin = 0.75*50*10
    acumBox, acumFill = scoreAccumulator(win, currentAmount, maxwin)
    acumBox.draw()
    acumFill.draw()
    drawlabels()
    win.flip()
    clock.reset()
    clicked = False
    while not clicked:
        keys = event.getKeys()
        if 'escape' in keys:
            core.quit()
        if 'z' in keys:
            rt = clock.getTime()
            _, value, name = bandits[0]
            if value == 0:
                loseWindow(win, "You lost!")
            else:
                winWindow(win, f"You won {value}")
            clicked = True
            return (value, rt, name)
        if 'm' in keys:
            rt = clock.getTime()
            _, value, name = bandits[1]
            if value == 0:
                loseWindow(win, "You lost!")
            else:
                winWindow(win, f"You won {value}")
            clicked = True
            return (value, rt, name)
        core.wait(0.01)
    return (0, 0, "")

def checkIfEscape():
    keys = event.getKeys()
    if 'escape' in keys:
        core.quit()

def loseWindow(win: visual.Window, message: str):
    winMsg = visual.TextStim(win, message, pos=(0, 0), color='white', height=50)
    winMsg.draw()
    win.flip()
    core.wait(0.5)

def winWindow(win: visual.Window, message: str):
    winMsg = visual.TextStim(win, message, pos=(0, 0), color='white', height=140)
    duration = 1
    blink_interval = 0.2
    timer = core.Clock()
    while timer.getTime() < duration:
        if int(timer.getTime() / blink_interval) % 2 == 0:
            winMsg.color = 'none'
            winMsg.draw()
            win.flip()
        else:
            winMsg.color = 'white'
            winMsg.draw()
            win.flip()

def scoreAccumulator(win: visual.Window, amount: int, maxAmount: float):
    boxheight = 700
    box = visual.Rect(
        win=win,
        width=75,
        height=boxheight,
        fillColor='none',
        lineColor='white',
        pos = (-650, 0)
    )
    fillHeight = (amount / maxAmount) * boxheight
    fill = visual.Rect(
        win=win,
        width=73,
        height=fillHeight,
        fillColor='yellow',
        lineColor='none',
        pos = (-650, -boxheight/2 + (fillHeight / 2))
    )
    return box, fill

def nbandits(win: visual.Window, distributions: list):
    bandits = []
    n_arms = len(distributions)
    for bandit in range(n_arms):
        value = calcWinning(distributions[bandit])
        name = distributions[bandit][1]
        xpos = (bandit*2 - n_arms/2) * (win.size[0] / 12)
        stim = visual.Rect(
            win=win,
            width=200,
            height=300,
            fillColor='white',
            lineColor='black',
            pos = (xpos, 0)
        )

        bandits.append((stim, value, name))
    return bandits

def floating_key_labels(win, left_stim, right_stim, base_offset=40, label_height=36):
    
    left_y  = left_stim.pos[1]  - (left_stim.height/2)  - base_offset
    right_y = right_stim.pos[1] - (right_stim.height/2) - base_offset

    left_label = visual.TextStim(
        win, text='z', height=label_height, color='white',
        pos=(left_stim.pos[0], left_y), alignText='center'
    )
    right_label = visual.TextStim(
        win, text='m', height=label_height, color='white',
        pos=(right_stim.pos[0], right_y), alignText='center'
    )

    def draw_labels():
        left_label.draw()
        right_label.draw()

    return draw_labels

def calcWinning(param):
    prob, _ = param
    exctWin = winAmount * np.random.binomial(n=1, p=prob)
    return exctWin

def slider(win: visual.Window, prompt: str, labels: tuple) -> float:
    slider = visual.Slider(
        win=win,
        size=(800, 100),
        pos=(0, 0),
        ticks=(0, 10),
        labels=labels,
        granularity=0,
        style='rating',
        color='LightGray',
        labelHeight=40,
        markerColor='DarkGray',
        lineColor='White'
    )
    promptText = visual.TextStim(win, text=prompt, pos=(0, 150), color='white', height=50)
    value = None

    while value is None:
        promptText.draw()
        slider.draw()
        win.flip()
        getkeys = event.getKeys(['return','enter', 'escape'])
        if 'escape' in getkeys:
            core.quit()
        if any(k in getkeys for k in ['return','enter']) and slider.getRating() is not None:
            value = slider.getRating()
    return value

def musicalBreak():
    breakText = visual.TextStim(win, text="Have a 10 second break for your ears and mind.", pos=(0, 0), color='white', height=50)
    breakText.draw()
    win.flip()
    core.wait(10)

def inductionTrial(file, skipable):
    introString = "Listen to the music and report the likability afterwards. \n" \
    "The music takes approx. 3 minutes."
    introText = visual.TextStim(win, text=introString, pos=(0, 0), color='white', height=60)
    introText.draw()
    win.flip()
    core.wait(2)
    introSound = sound.Sound(file)
    introSound.play()
    if skipable:
        event.waitKeys()
        introSound.stop()
    else:
        core.wait(introSound.getDuration())

    prompt = "Rate the music on the slider and press enter"
    answer = slider(win, prompt, ("Dislike", "Like"))
    return answer

def moodTest(win: visual.Window):
    prompt = "How happy do you feel at this moment? Rate on the slider and press enter"
    answer = slider(win, prompt, ("Unhappy", "Happy"))
    return answer

def controlTrial(file, skipable):
    introString = "Listen to the music and report the likability afterwards. \n" \
    "The music takes approx. 3 minutes."
    introText = visual.TextStim(win, text=introString, pos=(0, 0), color='white', height=60)
    introText.draw()
    win.flip()
    core.wait(2)
    introSound = sound.Sound(file)
    introSound.play()
    if skipable:
        event.waitKeys()
    else:
        core.wait(introSound.getDuration())

    prompt = "Rate the music on the slider and press enter"
    answer = slider(win, prompt, ("Dislike", "Like"))
    return answer

def instructionText(win: visual.Window):
    text1= ("Welcome to the experiment! \n" \
    "This experiment consists of several pieces of music needing your rating, some mood ratings and a decision-making game.\n"
    "You will play one practice round and then 8 blocks of the main game, every second preceded by a piece of music.\n"
    "Press any key to continue to the instructions.")
    instrText = visual.TextStim(win, text=text1, pos=(0, 0), color='white', height=30, wrapWidth=1000)
    instrText.draw()
    win.flip()
    event.waitKeys()

    text2= (
    "In the game, you will be presented with two white cards, and behind them there will either be a reward, or nothing\n"
    "Your task is to choose one of the cards by pressing the 'z' key for the left card, and the 'm' key for the right card.\n"
    "This will trigger a message, saying whether you won or not.\n" 
    "Your job is to maximize your points throughout the trials. \n" \
    "Use the information from previous trials to guide your choices!\n" \
    "Some cards will be more rewarding than others over time.\n" \
    "Press any key to start the experiment, good luck!")
    instrText = visual.TextStim(win, text=text2, pos=(0, 0), color='white', height=30, wrapWidth=1000)
    instrText.draw()
    win.flip()
    event.waitKeys()

def outroText(win: visual.Window):
    text= ("You have reached the end of the experiment! \n" \
    "Thank you very much for your participation. \n Have a dumle and a high-five on the way out!")
    outroText = visual.TextStim(win, text=text, pos=(0, 0), color='white', height=30, wrapWidth=1000)
    outroText.draw()
    win.flip()
    event.waitKeys()

def blockEval(amount, maxWin):
    if amount > maxWin:
        evalText = visual.TextStim(win, text=f"New highscore of {amount}!!", pos=(0, 0), color='white', height=100)
    else:
        evalText = visual.TextStim(win, text=f"The total score of this block is {amount}!", pos=(0, 0), color='white', height=50)
    evalText.draw()
    win.flip()
    core.wait(2)

def derivedInfo(results, trialInfo):
    numTrials, arm, winning = trialInfo
    
    if numTrials > 1:
        # FID 0, blockType 1, block 2, trial (1 indexed) 3, name 4, greedyChoice 5, winning 6, current 7, prevArm 8, switch 9, switchRate 10, armAProb 11, armBProb 12, armAcount 13, armBcount 14, armAval 15, armBval 16, rt 17
        last = results[-1]
        trial = last[3]
        prevArm = last[4]
        current = last[7]
        switchRate = last[10]
        armAcount = last[13]
        armBcount = last[14]
        armAval = last[15]
        armBval = last[16]

        greedyChoice = 'A' if (armAval > armBval) else 'B' if (armAval < armBval) else 'arb'

        current += winning
        if arm == 'A':
            armAcount += 1
            armAval += (1/(armAcount))*(winning - armAval)
        elif arm == 'B':
            armBcount += 1
            armBval += (1/(armBcount))*(winning - armBval)

        switch = 1 if (prevArm != arm) else 0
        switchRate += (1/(trial))*(switch - switchRate)
    else:
        greedyChoice = 'arb'
        armAcount = 0
        armBcount = 0
        armAval = 0
        armBval = 0
        prevArm = None 
        switch = None 
        switchRate = 0
        current = winning
        if arm == 'A':
            armAcount += 1
            armAval = winning
        elif arm == 'B':
            armBcount += 1
            armBval = winning

    return greedyChoice, armAcount, armBcount, armAval, armBval, prevArm, switch, switchRate, current

def trialBlock(win: visual.Window, distributions: list, ms: event.Mouse, n_trials: int, blockInfo: tuple):
    #Intro ting
    FID, blockType, block, highscore = blockInfo
    introText = visual.TextStim(win, text=f"Block {block}: Press any key to start", pos=(0, 0), color='white', height=60)
    introText.draw()
    win.flip()
    event.waitKeys()

    #Actual trials
    current = 0
    armAprob = max(distributions[0][0], distributions[1][0])
    armBprob = min(distributions[0][0], distributions[1][0])

    results = []
    for i in range(n_trials):
        winning, rt, name = trial(win, distributions, current, ms)
        greedyChoice, armAcount, armBcount, armAval, armBval, prevArm, switch, switchRate, current = derivedInfo(results, (i+1, name, winning))
        results.append((FID, blockType, block, i+1, name, greedyChoice, winning, current, prevArm, switch, switchRate, armAprob, armBprob, armAcount, armBcount, armAval, armBval, rt))

    blockEval(current, highscore)

    return (pd.DataFrame(results, columns=['FID', 'Blocktype','Block', 'Trial', 'Arm', 'GreedyChoice', 'Reward', 'Acummulated', 'PrevArm','Switch', 'SwitchRate', 'Arm-A', 'Arm-B', 'Arm-ACount', 'Arm-BCount', 'Arm-AVal', 'Arm-BVal', 'RT']), max(current, highscore))

def inductionBlocks(n_blocks: int, blockType: tuple, savePath: str, blockDistributions: list, trials_per_block: int, FID: int, latinorder: int, preMoodScore, highscore: int = 0, skipable: bool = False):
    #Concrete induction of musical vibe
    btype, inductionMusic, backgroundMusic = blockType
    if not (btype == "ctrl"):
        inductionTest = inductionTrial(inductionMusic, skipable)
    else:
        inductionTest = controlTrial(inductionMusic, skipable)

    #Background music
    if not (btype == "ctrl"):
        bgm = sound.Sound(backgroundMusic, loops=-1)
        bgm.play()

    #Actual bandit blocks
    newhighscore = highscore
    blockResultList = []

    for i in range(n_blocks):
        blockResults, newhighscore = trialBlock(win, blockDistributions[i], mouse, trials_per_block, (FID, btype, i+1, newhighscore))
        blockResultList.append(blockResults)

    #stop musikken
    if not (btype == "ctrl"):
        bgm.stop()

    postMoodScore = moodTest(win)

    resultPath = savePath + f'/results.csv'
    for blockResult in blockResultList:
        blockResult['PreMoodScore'] = preMoodScore
        blockResult['PostMoodScore'] = postMoodScore
        blockResult['InductionMusicScore'] = inductionTest
        blockResult['BlockOrder'] = latinorder
        blockResult.to_csv(resultPath, index=False, mode='a', header=not os.path.exists(resultPath))

    musicalBreak()
    
    return newhighscore, postMoodScore

def practiceBlock(savePath: str, blockDistributions: list, trials_per_block: int, FID: int):

    btype = 'Practice'
    practiceBlockResult, _ = trialBlock(win, blockDistributions, mouse, trials_per_block, (FID, btype, btype, 0))

    resultPath = savePath + f'/results.csv'
    practiceBlockResult['PreMoodScore'] = 'Practice'
    practiceBlockResult['PostMoodScore'] = 'Practice'
    practiceBlockResult['InductionMusicScore'] = 'Practice'
    practiceBlockResult['BlockOrder'] = 'Practice'
    practiceBlockResult.to_csv(resultPath, index=False, mode='a', header=not os.path.exists(resultPath))

    text= ("You have completed the practice round! \n" \
    "Press any key to continue to the main experiment.")
    outroText = visual.TextStim(win, text=text, pos=(0, 0), color='white', height=30, wrapWidth=1000)
    outroText.draw()
    win.flip()
    event.waitKeys()

def getInductionList(blockOrder: int):
    musicalInductionLists = [
    ("LaLv", "music/LaLvIntro.wav", "music/LaLvBG.wav"),
    ("HaLv", "music/HaLvIntro.wav", "music/HaLvBG.wav"),
    ("HaHv", "music/HaHvIntro.wav", "music/HaHvBG.wav"),
    ("LaHv", "music/LaHvIntro.wav", "music/LaHvBG.wav"),
    ("ctrl", 'music/CtrlIntro.wav', None),
    ]
    for i in range(blockOrder):
        musicalInductionLists.append(musicalInductionLists.pop(0))
    return musicalInductionLists

#Iitializing stuff
id, blockOrder = getSubjectInfo()
age, gender = getSubjectCharacteristics()

win = visual.Window(size=(800, 600), color='blue', units='pix', fullscr=True)
clock = core.Clock()
mouse = event.Mouse(visible=True, win=win)

musicalInduction = getInductionList(blockOrder)

winAmount = 10

distributions = [
    [[(0.75, 'A'), (0.55,'B')],
     [(0.60, 'A'), (0.50,'B')]],
    [[(0.75, 'A'), (0.55,'B')],
     [(0.60, 'A'), (0.50,'B')]],
    [[(0.75, 'A'), (0.55,'B')],
     [(0.60, 'A'), (0.50,'B')]],
    [[(0.75, 'A'), (0.55,'B')],
     [(0.60, 'A'), (0.50,'B')]],
    [[(0.75, 'A'), (0.55,'B')],
     [(0.60, 'A'), (0.50,'B')]],
    ]

for dist in distributions:
    for param in dist:
        np.random.shuffle(param)
    

#Actual control flow of experiment eller noget
os.makedirs(f'data/{id}', exist_ok=True)

if os.path.exists(f'data/{id}/results.csv'):
    os.remove(f'data/{id}/results.csv')

if os.path.exists(f'data/{id}/characteristics.csv'):
    os.remove(f'data/{id}/characteristics.csv')

characteristicsPath = f'data/{id}/characteristics.csv'
pd.DataFrame([(id, age, gender, blockOrder)], columns=['FID', 'Age', 'Gender', 'BlockOrder']).to_csv(characteristicsPath, index=False, mode='a', header=not os.path.exists(characteristicsPath))


permhighscore = 0
practiceDistrubutions = [(0.80, 'A'), (0.10,'B')]
n_practiceTrials = 5

'''
Det er dem herunder der skal ændres ift. test og det rigtige eksperiment!!!!!
'''
skipableMusic = False
trials_per_block = 50


instructionText(win)
practiceBlock(f'data/{id}', practiceDistrubutions, n_practiceTrials, id)

pms = moodTest(win)
for i, dist in enumerate(distributions):
    permhighscore, pms = inductionBlocks(2, musicalInduction[i], f'data/{id}', dist, trials_per_block, id, blockOrder, pms, permhighscore, skipableMusic)

outroText(win)
win.close()
