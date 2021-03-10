import wx
from bert_predictor import BertFilter
from MC_bert_predictor import MCBertFilter
from test_generator import TestGenerator


class MyFrame(wx.Frame):    
    def __init__(self):
        super().__init__(parent=None, title='Cloze test generator')
        panel = wx.Panel(self)        
        my_sizer = wx.BoxSizer(wx.VERTICAL)        
        self.text_ctrl = wx.TextCtrl(panel)
        my_sizer.Add(self.text_ctrl, 0, wx.ALL | wx.EXPAND, 5)        
        my_btn = wx.Button(panel, label='Generate tests')
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)        
        panel.SetSizer(my_sizer)        
        self.sent_filter = BertFilter()
        self.sent_generator = TestGenerator(self.sent_filter)
        self.sent_generator.read_corpus('sent_corpus.csv')
        self.instance_text = []
        
        
        self.Show()

    def on_press(self, event):
        value = self.text_ctrl.GetValue()
        for st in self.instance_text:
            st.Destroy()

        self.instance_text = []            
            
        if not value:
            print("You didn't enter any words!")
        else:         
            keywords = value.strip().split(' ')
            description = 'Cloze tests for: ' + value
            
            #st = wx.StaticText(self, label=description, pos=(5,65), style=wx.ALIGN_LEFT)
            #self.instance_text.append(st)
            
            exercises = []
            for word in keywords:
                exercise_rows = []
                exercise = self.sent_generator.generate_test(word, keywords)
                words = exercise.split(' ')
                row_str = words[0]
                for word in words[1:]:
                    if len(row_str + ' ' + word) < 60:  #Does adding the next word cause overflow? 
                        row_str += ' ' + word
                    else:
                        exercise_rows.append(row_str)
                        row_str = word
                exercise_rows.append(row_str)
                
                exercises.append(exercise_rows)
            
            
            row_placement = 65  
            for exercise in exercises:        
                for row in exercise:
                    st = wx.StaticText(self, label=row, pos=(5, row_placement), style=wx.ALIGN_LEFT)
                    row_placement += 15
                    self.instance_text.append(st)
                row_placement += 10


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
    
    
    
    
    
    
    
    
    
    
    