import mysql from "mysql2";
import dotenv from "dotenv";
dotenv.config();

const connection = mysql.createConnection({
  host: process.env.HOST || "localhost:3306",
  user: process.env.USER || "laravel_user",
  password: process.env.PASSWORD || "secure_passowrd",
  database: process.env.DATABASE || "fastich",
});

export default connection;
