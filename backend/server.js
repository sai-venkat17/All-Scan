const app=require("./app.js");
const dotenv=require('dotenv');
const mongoose=require("mongoose");
dotenv.config({path:"my_env.env"});
mongoose.connect(process.env.MONGO_URI,
{ useNewUrlParser: true, useUnifiedTopology: true }, err => {
        console.log('connected')
});
app.listen(process.env.SERVER_PORT,()=>{
    console.log(`Back-end running at ${process.env.SERVER_PORT}`);
})





