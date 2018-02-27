REM Edward Bujak
REM Deep Learning (BBDS)
REM Deep Learning Feed Forward Neural Network
REM Week 2 - February 2018

cd C:\tmp
REM del project1\*

REM Enter
REM http://DESKTOP-JLKM8MS:6006
REM or
REM http://DESKTOP-JLKM8MS:6006/#scalars&_smoothingWeight=0.905
REM in browser, to view:
REM     accuracy vs step_count
REM     loss vs step_count

REM Python 3.6 does NOT work with TensorFlow
REM "C:\Users\Edward Bujack\AppData\Local\Programs\Python\Python36\Scripts\tensorboard.exe" --logdir="project1"
REM Python 3.5 does work with TensorFlow
"C:\Users\Edward Bujack\AppData\Roaming\Python\Python35\Scripts\tensorboard.exe" --logdir="project1"

REM Alternative
REM C:\tmp>python -m tensorflow.tensorboard --logdir="project1"
REM but get
REM C:\Users\Edward Bujack\AppData\Local\Programs\Python\Python35\python.exe: No module named tensorflow.tensorboard

REM Why does this NOT work?
REM "C:\Users\Edward Bujack\AppData\Local\Programs\Python\Python35\python.exe" -m tensorflow.tensorboard --logdir="project1"
REM pause