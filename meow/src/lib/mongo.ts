import {MongoClient} from 'mongodb';

// TODO: generate these values using .env

const client = new MongoClient('mongodb+srv://william:123@cluster0.2nxfqjy.mongodb.net/?retryWrites=true&w=majority')

export function start_mongo() {
	console.log('Starting mongo...');
	return client.connect();
}

export const insert = client.db('mongodbVSCodePlaygroundDB').collection('images')

export default client.db('mongodbVSCodePlaygroundDB')