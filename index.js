import express from "express";
import cors from "cors";
import langRouter from "./routes/LangRouter.js";
import dotenv from "dotenv";
// import connection from "./config/db.js";

const PORT = process.env.PORT || 3001;

const app = express();

// connection.connect(function (err) {
//   if (err) throw err;
//   console.log("Database connected successfully");
// });
dotenv.config();
app.use(express.json());
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

app.options("*", cors());

app.get("/", (req, res) => {
  res.send("Hello World!");
});

app.use("/", langRouter);

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
