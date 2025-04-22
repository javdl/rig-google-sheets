use std::io::stdin;

use mcp_core::{
    client::ClientBuilder,
    transport::{ClientSseTransport, ClientSseTransportBuilder, ClientStdioTransport},
    types::{Implementation, ToolsListResponse},
};
use rig::{
    OneOrMany,
    completion::{CompletionModel, CompletionRequestBuilder, ToolDefinition},
    message::{AssistantContent, Message, ToolResult, ToolResultContent, UserContent},
    providers,
    tool::{McpTool, ToolSet},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp_client = connect_to_gsheets_mcp().await?;

    let tools_list_res = mcp_client.list_tools(None, None).await?;

    let (tools, tooldefs) = get_tools_from_mcp_tool_response(tools_list_res, mcp_client);

    let openai_client = providers::openai::Client::from_env();
    let model = openai_client.completion_model("gpt-4o");

    println!("Hi! How can I help you today? (write \"quit\" to exit)");
    println!("------------");

    let mut chat_history = Vec::new();

    loop {
        let prompt = take_input();
        println!("------------");

        if prompt == *"quit" {
            println!("Thanks for using me! I am quitting now.");
            break;
        }

        let res = call_until_response(
            prompt.into(),
            &model,
            PREAMBLE,
            &mut chat_history,
            &tools,
            tooldefs.clone(),
        )
        .await
        .unwrap();

        println!("{res}");
        println!("------------");
    }

    Ok(())
}

fn take_input() -> String {
    let mut str = String::new();
    stdin().read_line(&mut str).unwrap();

    str
}

async fn connect_to_gsheets_mcp()
-> Result<mcp_core::client::Client<ClientSseTransport>, Box<dyn std::error::Error>> {
    println!("Loading GSheets MCP server...");

    let client_transport =
        ClientSseTransportBuilder::new("http://127.0.0.1:3000/sse".to_string()).build();

    let mcp_client = ClientBuilder::new(client_transport).build();

    mcp_client.open().await?;

    mcp_client
        .initialize(
            Implementation {
                name: "echo".to_string(),
                version: "1.0".to_string(),
            },
            mcp_core::types::ClientCapabilities::default(),
        )
        .await?;

    println!("Successfully opened.");

    Ok(mcp_client)
}

fn get_tools_from_mcp_tool_response(
    tools_list_res: ToolsListResponse,
    mcp_client: mcp_core::client::Client<ClientSseTransport>,
) -> (ToolSet, Vec<ToolDefinition>) {
    let (tools, tooldefs) = tools_list_res.tools.into_iter().fold(
        (ToolSet::builder().build(), Vec::new()),
        |(mut tools, mut tooldefs), tool| {
            let mcp_tool = McpTool::from_mcp_server(tool.clone(), mcp_client.clone());
            tools.add_tool(mcp_tool);

            let tooldef = ToolDefinition {
                description: tool.description.unwrap_or_default(),
                name: tool.name,
                parameters: tool.input_schema.clone(),
            };
            tooldefs.push(tooldef);

            (tools, tooldefs)
        },
    );

    (tools, tooldefs)
}
const PREAMBLE: &str = r###"You are an agent designed to qualify sales leads from Google Sheets.

Users will typically ask you to qualify leads from Google Forms submissions
(or imported spreadsheets from results of other form submission-type applications).

Your job is to qualify sales leads based on the user's criteria.
If they don't give you a criteria for qualification,
ask what demographic the user is trying to capture with the form and qualify leads based off of that.

When creating the results, use a new sheet in the spreadsheet file the user has provided you with.
When done, specify the location of the sheet so that the user can inspect the result for themselves.
"###;

async fn call_until_response<M: CompletionModel>(
    mut prompt: Message,
    model: &M,
    preamble: &str,
    chat_history: &mut Vec<Message>,
    toolset: &ToolSet,
    tooldefs: Vec<ToolDefinition>,
) -> Result<String, anyhow::Error> {
    loop {
        let request = CompletionRequestBuilder::new(model.clone(), prompt.to_owned())
            .preamble(preamble.to_owned())
            .messages(chat_history.clone())
            .temperature(0.0)
            .max_tokens(1024)
            .tools(tooldefs.clone())
            .build();
        // call model
        let resp = model
            .completion(request)
            .await
            .map_err(|x| anyhow::anyhow!("Error when prompting: {x}"))?;

        // keep calling tools until we get human readable answer from the model
        match resp.choice.first() {
            AssistantContent::Text(text) => {
                let text = text.text;
                chat_history.push(prompt.clone());
                chat_history.push(Message::assistant(&text));
                return Ok(text);
            }
            AssistantContent::ToolCall(tool_call) => {
                // Call the tool
                let tool_response = toolset
                    .call(
                        &tool_call.function.name,
                        tool_call.function.arguments.to_string(),
                    )
                    .await;

                let tool_response = match tool_response {
                    Ok(res) => res,
                    Err(e) => {
                        chat_history.push(prompt.clone());
                        chat_history.push(Message::Assistant {
                            content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
                        });
                        prompt = Message::User {
                            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                                id: tool_call.id.to_string(),
                                content: OneOrMany::one(ToolResultContent::Text(
                                    rig::message::Text {
                                        text: e.to_string(),
                                    },
                                )),
                            })),
                        };
                        continue;
                    }
                };

                let tool_response_message = UserContent::tool_result(
                    tool_call.id.clone(),
                    OneOrMany::one(ToolResultContent::Text(tool_response.into())),
                );

                let tool_call = OneOrMany::one(AssistantContent::ToolCall(tool_call));

                // add tool call and response into chat history and continue the loop
                chat_history.push(prompt.clone());
                chat_history.push(Message::Assistant { content: tool_call });

                let tool_result_message = Message::User {
                    content: OneOrMany::one(tool_response_message),
                };

                prompt = tool_result_message;
            }
        }
    }
}
