const express=require('express');
const bodyParser=require('body-parser');
const cors=require("cors");
const imgModel=require("./model");
const ejs=require("ejs");
const app=express();
const FormData=require('form-data');
const request=require("request");
const Dropbox=require("dropbox");
const upload=require('express-fileupload');
const buffer=require('buffer');
const fs=require('fs')
const { time } = require('console');
const { syncBuiltinESMExports } = require('module');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
app.use(cors({
    origin:"*"
}))
app.use(upload());
app.set("view engine","ejs");
app.set("views","frontend");
app.use( express.static( "inter" ) );
app.use(express.json());
app.use(bodyParser.urlencoded({extended:false}));

    
app.get("/",(req,res)=>{
    res.render('homepage.ejs');
})

app.get("/home",(req,res)=>{
    res.render('homepage.ejs');
})

app.get("/about",(req,res)=>{
    res.render('about.ejs');
})
var api="http://127.0.0.1:5000/"

app.get("/brain",(req,res)=>{
    var options = {
        'method': 'POST',
        'url': api+'brain',
        'headers': {
            'Content-Type': 'application/json'
        },
        formData: {
          'file_name': "brain_model.jpg"
        }
    }
        request(options,function(error,response){
            if(error) throw new Error(error);
            console.log(response.body);
            res.render("analysis_page",{res:response.body})
        })
})

app.get("/lung",(req,res)=>{
    var options = {
        'method': 'POST',
        'url': 'http://127.0.0.1:5000/lung',
        'headers': {
            'Content-Type': 'application/json'
        },
        formData: {
          'file_name': "lung_model.jpg"
        }
    }
        request(options,function(error,response){
            if(error) throw new Error(error);
            console.log(response.body);
            res.render("analysis_page",{res:response.body})
        })
})

app.get("/kidney",(req,res)=>{
    var options = {
        'method': 'POST',
        'url': 'http://127.0.0.1:5000/brain',
        'headers': {
            'Content-Type': 'application/json'
        },
        formData: {
          'file_name': "kidney_model.jpeg"
        }
    }
        request(options,function(error,response){
            if(error) throw new Error(error);
            console.log(response.body);
            imgModel.findOne({"name":"img1"})
                        .then(function(resp){
                            fs.writeFileSync("inter2/img"+1+".jpg",resp.img.data)
                            imgModel.findOne({"name":"img2"})
                            .then(function(resp){
                                fs.writeFileSync("inter2/img"+2+".jpg",resp.img.data)
                                imgModel.findOne({"name":"img5"})
                                .then(function(resp){
                                    fs.writeFileSync("inter2/img5.jpg",resp.img.data)
                                    res.render("analysis_page",{res:response.body})
                                })  
                            })
                        })
                        .catch(function(err){
                            console.log(err);
                        })
        })
})






app.post("/analyze",async(req,res)=>{
    if(req.body.Scan_type=="brain"){
  
    if(req.files){
      
      const uploaded_img=req.files.filename;
      const uploaded_img_name=uploaded_img.name;
      //upload_file(uploaded_img)
        // var options = {
        //     'method': 'POST',
        //     'url': 'http://127.0.0.1:5000/brain',
        //     'headers': {
        //         'Content-Type': 'application/json'
        //     },
        //     formData: {
        //       'file_name': uploaded_img_name
        //     }
        // }
        // request(options,function(error,response){
        //     if(error) throw new Error(error);
        //     console.log(response.body);
        //     res.render("analysis_page",{res:response.body})
        // })
      
      
    //}
      uploaded_img.mv(__dirname.substring(0,__dirname.length-7)+"/uploads/"+uploaded_img_name,(err)=>{
          if(err){
              console.log("File upload error:"+err);
          }
          else{
            console.log("Image got uploaded to server")
            var options = {
                'method': 'POST',
                'url': 'http://127.0.0.1:5000/brain',
                'headers': {
                    'Content-Type': 'application/json'
                },
                formData: {
                  'file_name': uploaded_img_name
                }
            }
            var obj={
                name:uploaded_img_name,
                img:{
                data: fs.readFileSync("uploads/"+uploaded_img_name),
                contentType: 'image/jpg'
                }
            }
            imgModel.deleteOne({"name":uploaded_img_name},()=>{
            imgModel.create(obj, (err, item) => {
                if (err) {
                    console.log(err);
                }
                else {
                    // item.save();
                    request(options,function(error,response){
                        if(error) throw new Error(error);
                        console.log(response.body);
                        imgModel.findOne({"name":"img1"})
                        .then(function(resp){
                            fs.writeFileSync("inter2/img"+1+".jpg",resp.img.data)
                            imgModel.findOne({"name":"img2"})
                            .then(function(resp){
                                fs.writeFileSync("inter2/img"+2+".jpg",resp.img.data)
                                imgModel.findOne({"name":"img5"})
                                .then(function(resp){
                                    fs.writeFileSync("inter2/img5.jpg",resp.img.data)
                                    res.render("analysis_page",{res:response.body})
                                })  
                            })
                        })
                        .catch(function(err){
                            console.log(err);
                        })
                        
                    
                    
                        
                    })    
                }
            });
        })
              };
              
          })
        
      }
      else{
        var options = {
            'method': 'POST',
            'url': 'http://127.0.0.1:5000/brain',
            'headers': {
                'Content-Type': 'application/json'
            },
            formData: {
              'file_name': "brain_model.jpg"
            }
        }
            request(options,function(error,response){
                if(error) throw new Error(error);
                console.log(response.body);
                res.render("analysis_page",{res:response.body})
            })
      }
    }
    else if(req.body.Scan_type=="lung"){
        if(req.files){
      
            const uploaded_img=req.files.filename;
            const uploaded_img_name=uploaded_img.name;
            uploaded_img.mv(__dirname.substring(0,__dirname.length-7)+"/uploads/"+uploaded_img_name,(err)=>{
                if(err){
                    console.log("File upload error:"+err);
                }
                else{
                  console.log("Image got uploaded to server")
                  var options = {
                      'method': 'POST',
                      'url': 'http://127.0.0.1:5000/lung',
                      'headers': {
                          'Content-Type': 'application/json'
                      },
                      formData: {
                        'file_name': uploaded_img_name
                      }
                  }
                      request(options,function(error,response){
                          if(error) throw new Error(error);
                          console.log(response.body);
                          res.render("analysis_lung",{res:response.body})
                      })
                    };
                    
                })
            }
            else{
                var options = {
                    'method': 'POST',
                    'url': 'http://127.0.0.1:5000/lung',
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    formData: {
                      'file_name': "lung_model.png"
                    }
                }
                    request(options,function(error,response){
                        if(error) throw new Error(error);
                        console.log(response.body);
                        res.render("analysis_lung",{res:response.body})
                    })
            }
    }
    else if(req.body.Scan_type=="kidney"){
        if(req.files){
      
            const uploaded_img=req.files.filename;
            const uploaded_img_name=uploaded_img.name;
            uploaded_img.mv(__dirname.substring(0,__dirname.length-7)+"/uploads/"+uploaded_img_name,(err)=>{
                if(err){
                    console.log("File upload error:"+err);
                }
                else{
                  console.log("Image got uploaded to server")
                  var options = {
                      'method': 'POST',
                      'url': 'http://127.0.0.1:5000/brain',
                      'headers': {
                          'Content-Type': 'application/json'
                      },
                      formData: {
                        'file_name': 'kidney_model.jpeg'
                      }
                  }
                      request(options,function(error,response){
                          if(error) throw new Error(error);
                          console.log(response.body);
                          res.render("analysis_lung",{res:response.body})
                      })
                    };
                    
                })
            }
            else{
                var options = {
                    'method': 'POST',
                    'url': 'http://127.0.0.1:5000/brain',
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    formData: {
                      'file_name': "kidney_model.jpeg"
                    }
                }
                    request(options,function(error,response){
                        if(error) throw new Error(error);
                        console.log(response.body);
                        res.render("analysis_page",{res:response.body})
                    })
            }
    }
})



    


module.exports=app;