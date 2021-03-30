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
        
        # Load filter and sentence generator
        sent_filter = BertFilter()
        corpus_file = 'complete_sent_corpus.csv'       
        self.sent_generator = TestGenerator(sent_filter, corpus_file)
        self.instance_text = []
        
        self.Show()

    def on_press(self, event):
        value = self.text_ctrl.GetValue()
        for st in self.instance_text:
            st.Destroy()

        self.instance_text = []            
            
        if not value:
            msg = "You didn't enter any words!"
            st = wx.StaticText(self, label=msg, pos=(5, 65), style=wx.ALIGN_LEFT)
            self.instance_text.append(st)
            
            
        else:         
            keywords = value.strip().split(' ')
            
            exercises = []
            for word in keywords:
                exercise_rows = []
                exercise = self.sent_generator.generate_test(word, keywords)
                
                # Split the exercise into a suitable amount of rows so the text fits 
                # in the GUI window
                words = exercise.split(' ')
                row_str = words[0]
                for word in words[1:]:
                    # Check if adding the next word causes overflow
                    if len(row_str + ' ' + word) < 60:  
                        row_str += ' ' + word
                    # If it does, add the row to the row-list and start a new one    
                    else:
                        exercise_rows.append(row_str)
                        row_str = word
                # Add the final row of the exercise     
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
    
    
    
    
    
    
    
    
    
    
    