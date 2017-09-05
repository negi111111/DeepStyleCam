//
//  ViewController.h
//  styletransferforipad
//
//  Created by Ryousuke Tanno on 2017/08/10.
//  Copyright © 2017年 Ryosuke Tanno. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

@interface ViewController : UIViewController <AVCaptureVideoDataOutputSampleBufferDelegate, UIPickerViewDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *previewView;

- (IBAction)weight_select1:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select1;
- (IBAction)weight_select2:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select2;
- (IBAction)weight_select3:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select3;
- (IBAction)weight_select4:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select4;
- (IBAction)weight_select5:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select5;
- (IBAction)weight_select6:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select6;
- (IBAction)weight_select7:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select7;
- (IBAction)weight_select8:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select8;
- (IBAction)weight_select9:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select9;
- (IBAction)weight_select10:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select10;
- (IBAction)weight_select11:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select11;
- (IBAction)weight_select12:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select12;
- (IBAction)weight_select13:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select13;
- (IBAction)reset:(id)sender;
- (IBAction)Random:(id)sender;

@property (weak, nonatomic) IBOutlet UIButton *reset;
@property (weak, nonatomic) IBOutlet UIButton *Random;
- (IBAction)image_quality_low:(id)sender;
@property (weak, nonatomic) IBOutlet UIButton *image_quality_low;
- (IBAction)image_quality_medium:(id)sender;
@property (weak, nonatomic) IBOutlet UIButton *medium;
- (IBAction)image_quality_high:(id)sender;
@property (weak, nonatomic) IBOutlet UIButton *image_quality_high;

- (IBAction)weight_select14:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select14;
- (IBAction)weight_select15:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select15;
- (IBAction)weight_select16:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select16;




- (IBAction)weight_select17:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select17;
- (IBAction)weight_select18:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select18;
- (IBAction)weight_select19:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select19;
- (IBAction)weight_select20:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select20;
- (IBAction)weight_select21:(id)sender;


@property (weak, nonatomic) IBOutlet UISlider *weight_select21;
- (IBAction)weight_select22:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select22;


- (IBAction)weight_select23:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select23;
- (IBAction)weight_select24:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select24;

- (IBAction)weight_select25:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select25;

- (IBAction)weight_select26:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select26;
- (IBAction)weight_select27:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select27;

- (IBAction)weight_select28:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select28;
- (IBAction)weight_select29:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select29;
- (IBAction)weight_select30:(id)sender;

@property (weak, nonatomic) IBOutlet UISlider *weight_select30;

- (IBAction)weight_select31:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select31;
- (IBAction)weight_select32:(id)sender;
@property (weak, nonatomic) IBOutlet UISlider *weight_select32;


@property (weak, nonatomic) IBOutlet UITextField *weight_value1;

@property (weak, nonatomic) IBOutlet UITextField *weight_value2;
@property (weak, nonatomic) IBOutlet UITextField *weight_value3;

@property (weak, nonatomic) IBOutlet UITextField *weight_value4;

@property (weak, nonatomic) IBOutlet UITextField *weight_value5;
@property (weak, nonatomic) IBOutlet UITextField *weight_value6;
@property (weak, nonatomic) IBOutlet UITextField *weight_value7;

@property (weak, nonatomic) IBOutlet UITextField *weight_value8;

@property (weak, nonatomic) IBOutlet UITextField *weight_value9;

@property (weak, nonatomic) IBOutlet UITextField *weight_value10;


@property (weak, nonatomic) IBOutlet UITextField *weight_value11;
@property (weak, nonatomic) IBOutlet UITextField *weight_value12;

@property (weak, nonatomic) IBOutlet UITextField *weight_value13;


@property (weak, nonatomic) IBOutlet UITextField *weight_value14;


@property (weak, nonatomic) IBOutlet UITextField *weight_value15;




@property (weak, nonatomic) IBOutlet UITextField *weight_value16;

@property (weak, nonatomic) IBOutlet UITextField *weight_value17;


@property (weak, nonatomic) IBOutlet UITextField *weight_value18;



@property (weak, nonatomic) IBOutlet UITextField *weight_value19;
@property (weak, nonatomic) IBOutlet UITextField *weight_value20;

@property (weak, nonatomic) IBOutlet UITextField *weight_value21;
@property (weak, nonatomic) IBOutlet UITextField *weight_value22;

@property (weak, nonatomic) IBOutlet UITextField *weight_value23;


@property (weak, nonatomic) IBOutlet UITextField *weight_value24;
@property (weak, nonatomic) IBOutlet UITextField *weight_value25;

@property (weak, nonatomic) IBOutlet UITextField *weight_value26;

@property (weak, nonatomic) IBOutlet UITextField *weight_value27;
@property (weak, nonatomic) IBOutlet UITextField *weight_value28;
@property (weak, nonatomic) IBOutlet UITextField *weight_value29;
@property (weak, nonatomic) IBOutlet UITextField *weight_value30;

@property (weak, nonatomic) IBOutlet UITextField *weight_value31;

@property (weak, nonatomic) IBOutlet UITextField *weight_value32;

- (IBAction)color:(id)sender;




- (IBAction)standard:(id)sender;

@end

