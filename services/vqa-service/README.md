# visual question answering service

Service currently exposes one method *answer*
    
It accepts VqaRequest message:

```
message VqaRequest {
    string question = 1; // text question
    bool use_pm = 3;  // if true use pattern matcher to compute answer
    bytes image_data = 4; // image
}
```

returns VqaResponse message

```
message VqaResponse {
    string answer = 1;    // answer
    bool ok = 2;          // if ok == false then there where error
    string error_message = 3;  // error description
}
```
